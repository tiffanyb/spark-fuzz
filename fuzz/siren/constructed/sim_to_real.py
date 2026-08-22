"""Replay a SIREN trace (render_stock.py --trace) on a real Unitree G1.

    dry run (DEFAULT -- no DDS, no motion, nothing is sent):
        python sim_to_real.py --trace ssa_0_1_trace.csv

    inspect the robot without commanding it:
        python sim_to_real.py --trace ssa_0_1_trace.csv --check --iface en7

    actually move the arms:
        python sim_to_real.py --trace ssa_0_1_trace.csv --go --iface en7

WHAT THIS SENDS
    The trace's q_* columns are joint POSITION targets, one row per control
    step. They are written to motor_cmd[real_index].q with a PD gain, at the
    rate the trace was recorded at. Cartesian columns (L_ee_*, R_ee_*) are NOT
    commanded -- the low-level SDK has no Cartesian interface, and re-solving IK
    from an end-effector position would pick a different elbow configuration for
    the same hand position, which is exactly what the trace exists to pin down.
    Use those columns to CHECK a replay, not to drive one.

READ THIS BEFORE --go
    These traces were searched for BECAUSE THEY COLLIDE. `contact_step` in the
    metadata is the step at which the arm intersects the obstacle. By default
    this script stops one step before it, and refuses to go further unless you
    pass --through-contact. Replaying the tail with the obstacle physically
    present drives the arm into it.

    Sign and offset conventions are NOT verified here. The q values come from
    SPARK's MuJoCo model; whether motor k on your machine uses the same zero and
    the same sign as joint k in that model is a property of your robot, not of
    this file. Confirm with a single-joint nudge (unitree/demo/inject/
    wrist_twist_wired.py --test-pulse) before replaying a whole trajectory.

    The robot should be hoisted. Taking low-level control releases the balance
    controller and the legs go limp.

    The WAIST (motors 12-14) is driven by default. That is what reproduces the
    recorded trajectory -- holding it at zero puts the hand ~180 mm off the
    trace -- but it means the torso of a hoisted robot moves, and the waist gain
    here is not hardware-validated. --no-waist reverts to arms-only, at the cost
    of fidelity.
"""

import argparse
import csv
import json
import math
import os
import sys
import time

RATE_DEFAULT = 100.0
N_MOTORS = 35
# G1 layout, matching unitree/demo/inject/wrist_twist_wired.py
WAIST_IDX = range(12, 15)
ARM_IDX = range(15, 29)
LEG_IDX = range(0, 12)
ARM_KP, ARM_KD, LEG_KD = 40.0, 1.0, 2.0
# The waist is driven by DEFAULT because holding it at zero is not faithful --
# every arm frame hangs off it, and on ssa_0_1 that moves the hand ~180 mm and
# turns a 0.09 mm graze into a 26 mm strike (measured, viz_dryrun.py).
# This gain is NOT validated on hardware: unitree/demo/inject/
# wrist_twist_wired.py only ever drove arms. It starts at the arm value; check
# it with a small nudge before trusting a whole trajectory. --waist-kp overrides.
WAIST_KP, WAIST_KD = 40.0, 1.0


def load(trace_path, meta_path=None):
    """Trace rows + the metadata sidecar that gives the numbers meaning."""
    if meta_path is None:
        stem = os.path.splitext(trace_path)[0]
        meta_path = stem + "_meta.json"
    if not os.path.exists(meta_path):
        raise SystemExit(
            f"metadata sidecar not found: {meta_path}\n"
            "It carries the joint order and the sim->motor mapping; the CSV's "
            "numbers cannot be addressed to motors without it. Re-render with "
            "--trace to produce it.")
    with open(trace_path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{trace_path} has no rows")
    return rows, json.load(open(meta_path))


def plan(rows, meta, include_waist, limit_margin):
    """Which motors to drive, and the per-step targets, with limits checked.

    Returns (joints, targets, notes) where joints is a list of dicts carrying
    both index spaces, and targets[i] maps real_index -> commanded radians.
    """
    joints, skipped, notes = [], [], []
    for j in meta["robot"]["joints"]:
        col = f"q_{j['dof']}"
        if col not in rows[0]:
            notes.append(f"trace has no column {col}; joint not driven")
            continue
        ri = j.get("real_index")
        if ri is None:
            notes.append(f"{j['dof']}: no real_index in metadata; not driven")
            continue
        if ri in ARM_IDX:
            pass
        elif ri in WAIST_IDX:
            if not include_waist:
                skipped.append(j["dof"])
                continue
        else:
            notes.append(f"{j['dof']} -> motor {ri} is outside arms/waist; "
                         "not driven")
            continue
        joints.append({"dof": j["dof"], "col": col, "real": int(ri),
                       "sim": j.get("sim_index"),
                       "lim": j.get("pos_limit"), "kp": j.get("kp"),
                       "kd": j.get("kd")})
    if skipped:
        notes.append(f"--no-waist: waist NOT driven ({', '.join(skipped)})")
        notes.append("!! NOT FAITHFUL: every arm frame hangs off the waist, so "
                     "holding it at 0 moves the")
        notes.append("   hand ~180 mm off the recorded path. Measured on "
                     "ssa_0_1 with viz_dryrun.py, this")
        notes.append("   makes the arm hit the obstacle ~26 mm DEEPER than the "
                     "trace, not miss it. Verify")
        notes.append("   with: python -m fuzz.siren.constructed.viz_dryrun "
                     "--trace <csv> --out <dir> --no-waist")
    else:
        notes.append("waist IS driven (motors 12-14) -- this reproduces the "
                     "recorded path exactly (0.00 mm),")
        notes.append("   but the waist gain is NOT hardware-validated and this "
                     "moves the torso of a hoisted")
        notes.append("   robot. Nudge-test it before a full replay; --no-waist "
                     "opts out.")
    if not joints:
        raise SystemExit("no drivable joints found; check the metadata")

    targets, viol = [], []
    for n, r in enumerate(rows):
        step = {}
        for j in joints:
            q = float(r[j["col"]])
            lo_hi = j["lim"]
            if lo_hi:
                lo, hi = float(lo_hi[0]) + limit_margin, float(lo_hi[1]) - limit_margin
                if q < lo or q > hi:
                    viol.append((n, j["dof"], q, lo_hi))
            step[j["real"]] = q
        targets.append(step)
    return joints, targets, notes, viol


def motion_stats(joints, targets, dt):
    """Per-joint excursion and the worst single-step jump, which is the number
    that decides whether a replay is gentle or violent."""
    out = {}
    for j in joints:
        seq = [t[j["real"]] for t in targets]
        d = [abs(b - a) for a, b in zip(seq, seq[1:])] or [0.0]
        out[j["dof"]] = {
            "real": j["real"], "min": min(seq), "max": max(seq),
            "range": max(seq) - min(seq), "max_step": max(d),
            "max_rate": max(d) / dt if dt else float("nan"),
        }
    return out


def report(rows, meta, joints, targets, stats, notes, viol, dt, stop, args):
    t = meta.get("timing", {})
    print(f"  trace      {os.path.basename(args.trace)}  {len(rows)} rows")
    print(f"  robot      {meta['robot']['class_name']}  "
          f"{meta['robot']['n_dof']} DoF of {meta['robot']['num_total_motors']} "
          f"motors  ({meta['robot']['control_mode']})")
    print(f"  rate       dt {dt*1000:.2f} ms = {1/dt:.1f} Hz"
          f"   (metadata says {t.get('rate_hz')} Hz)")
    print(f"  contact    step {t.get('contact_step')}   replaying rows "
          f"0..{stop-1}" + ("" if args.through_contact else
                            "  (stopping before contact)"))
    print(f"  driving    {len(joints)} motors: "
          f"{', '.join(str(j['real']) for j in joints)}")
    for n in notes:
        print(f"  note       {n}")
    print()
    print(f"  {'dof':<22}{'motor':>6}{'min':>9}{'max':>9}{'range':>9}"
          f"{'max/step':>10}{'rad/s':>9}")
    for k, v in stats.items():
        print(f"  {k:<22}{v['real']:>6}{v['min']:>9.4f}{v['max']:>9.4f}"
              f"{v['range']:>9.4f}{v['max_step']:>10.5f}{v['max_rate']:>9.3f}")
    worst = max(stats.values(), key=lambda v: v["max_rate"])
    print(f"\n  fastest joint {worst['max_rate']:.3f} rad/s "
          f"({worst['max_rate']*57.3:.1f} deg/s)")
    if viol:
        print(f"\n  !! {len(viol)} position-limit violations, first 5:")
        for n, d, q, lim in viol[:5]:
            print(f"     row {n:>5} {d:<22} q={q:+.4f} outside {lim}")
    else:
        print("  position limits: all rows inside RealMotorPosLimit"
              f" (margin {args.limit_margin} rad)")


def dry_run(rows, joints, targets, stop, args):
    n_show = min(args.show, stop)
    print(f"\n  === DRY RUN: nothing is sent. First {n_show} and last 3 "
          f"commands ===")
    idx = list(range(n_show)) + ([None] + list(range(max(0, stop - 3), stop))
                                 if stop > n_show else [])
    for i in idx:
        if i is None:
            print("  ...")
            continue
        r = rows[i]
        parts = "  ".join(f"m{j['real']}={targets[i][j['real']]:+.4f}"
                          for j in joints[:6])
        more = "" if len(joints) <= 6 else f"  (+{len(joints)-6} more)"
        print(f"  t={float(r['t_s']):6.3f}s leg={r['leg']} "
              f"clr={float(r['clearance']):+.6f}  {parts}{more}")
    if args.dump:
        with open(args.dump, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "t_s"] + [f"m{j['real']}_{j['dof']}"
                                          for j in joints])
            for i in range(stop):
                w.writerow([i, rows[i]["t_s"]]
                           + [f"{targets[i][j['real']]:.9f}" for j in joints])
        print(f"\n  full command stream -> {args.dump}")
    print("\n  Nothing was sent. Re-run with --go (and --iface) to move the "
          "robot.")


def live(rows, meta, joints, targets, stop, dt, args):
    """Send the trace over rt/lowcmd. Mirrors the takeover sequence in
    unitree/demo/inject/wrist_twist_wired.py, which is the pattern already
    known to work on this machine."""
    try:
        from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                                 ChannelPublisher,
                                                 ChannelSubscriber)
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (LowCmd_, LowState_,
                                                            MotorCmd_)
        from unitree_sdk2py.utils.crc import CRC
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client \
            import MotionSwitcherClient
    except ImportError as e:
        raise SystemExit(f"unitree_sdk2py not importable ({e}).\n"
                         "Run the dry run instead, or use the environment that "
                         "has the SDK installed.")

    print(f"\n  init DDS on {args.iface}")
    ChannelFactoryInitialize(0, args.iface)
    msc = MotionSwitcherClient(); msc.SetTimeout(5.0); msc.Init()
    code, mode = msc.CheckMode()
    print(f"  CheckMode -> code={code} mode={mode}")

    sub = ChannelSubscriber("rt/lowstate", LowState_); sub.Init()

    if args.check:
        st = sub.Read(500)
        if st is None:
            print("  no rt/lowstate. Is the robot on and the iface right?")
            return 1
        print(f"  mode_machine={st.mode_machine}")
        print(f"  {'dof':<22}{'motor':>6}{'measured q':>12}{'trace q[0]':>12}"
              f"{'delta':>10}")
        worst = 0.0
        for j in joints:
            q = st.motor_state[j["real"]].q
            tgt = targets[0][j["real"]]
            worst = max(worst, abs(tgt - q))
            print(f"  {j['dof']:<22}{j['real']:>6}{q:>12.4f}{tgt:>12.4f}"
                  f"{tgt-q:>+10.4f}")
        print(f"\n  largest gap to the trace's first pose: {worst:.4f} rad "
              f"({worst*57.3:.1f} deg) -- the approach ramp has to cover this")
        print("  --check only: nothing was sent.")
        return 0

    active = bool(mode) and mode.get("name")
    if active:
        print(f"  releasing high-level mode {active!r} (limbs go limp -- the "
              "robot must be hoisted)")
        rc, _ = msc.ReleaseMode(); print(f"  ReleaseMode -> code={rc}")
        time.sleep(1.2)

    st, t0 = None, time.monotonic()
    while time.monotonic() - t0 < 5:
        st = sub.Read(100)
        if st is not None and any(abs(st.motor_state[i].q) > 1e-4
                                  for i in range(29)):
            break
    if st is None or not any(abs(st.motor_state[i].q) > 1e-4 for i in range(29)):
        raise SystemExit("  rt/lowstate absent or all-zero -> low-level not "
                         "active; aborting without sending anything.")
    mm = st.mode_machine
    q_now = {i: st.motor_state[i].q for i in range(29)}
    gap = max(abs(targets[0][j["real"]] - q_now[j["real"]]) for j in joints)
    print(f"  lowstate live, mode_machine={mm}, largest gap to row 0: "
          f"{gap:.4f} rad ({gap*57.3:.1f} deg)")
    if gap > args.max_approach:
        raise SystemExit(
            f"  gap {gap:.4f} rad exceeds --max-approach {args.max_approach}. "
            "The arm would swing hard to reach the trace's first pose. Move it "
            "closer by hand, or raise the limit deliberately.")

    pub = ChannelPublisher("rt/lowcmd", LowCmd_); pub.Init()
    crc = CRC()

    def send(q_by_motor, stiff):
        motors = [MotorCmd_(mode=0, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0,
                            reserve=0) for _ in range(N_MOTORS)]
        for i in LEG_IDX:
            motors[i] = MotorCmd_(mode=1, q=0.0, dq=0.0, tau=0.0, kp=0.0,
                                  kd=LEG_KD, reserve=0)
        if not args.include_waist:
            for i in WAIST_IDX:
                motors[i] = MotorCmd_(mode=1, q=0.0, dq=0.0, tau=0.0, kp=0.0,
                                      kd=LEG_KD, reserve=0)
        for j in joints:
            i = j["real"]
            waist = 12 <= i <= 14
            motors[i] = MotorCmd_(
                mode=1, q=float(q_by_motor[i]), dq=0.0, tau=0.0,
                kp=(args.waist_kp if waist else ARM_KP) * stiff,
                kd=(WAIST_KD if waist else ARM_KD), reserve=0)
        cmd = LowCmd_(mode_pr=0, mode_machine=mm, motor_cmd=motors,
                      reserve=[0, 0, 0, 0], crc=0)
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)

    try:
        print(f"  taking control: stiffness 0 -> 1 over {args.takeover}s at "
              "the current pose")
        hold = {j["real"]: q_now[j["real"]] for j in joints}
        n = max(1, int(args.takeover * (1 / dt)))
        for k in range(n + 1):
            send(hold, k / n); time.sleep(dt)

        print(f"  approaching row 0 over {args.approach}s")
        n = max(1, int(args.approach * (1 / dt)))
        for k in range(n + 1):
            f = k / n
            send({j["real"]: hold[j["real"]]
                  + (targets[0][j["real"]] - hold[j["real"]]) * f
                  for j in joints}, 1.0)
            time.sleep(dt)

        print(f"  replaying {stop} steps ({stop*dt:.2f}s)")
        t_next = time.monotonic()
        for i in range(stop):
            send(targets[i], 1.0)
            t_next += dt
            slack = t_next - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            if i % int(1 / dt) == 0:
                s = sub.Read(0)
                if s is not None:
                    j0 = joints[0]
                    print(f"   t={i*dt:5.2f}s  m{j0['real']} "
                          f"cmd={targets[i][j0['real']]:+.4f} "
                          f"meas={s.motor_state[j0['real']].q:+.4f}")
        print("  holding final pose")
        for _ in range(int(0.4 / dt)):
            send(targets[stop - 1], 1.0); time.sleep(dt)
    finally:
        print("  releasing: stiffness -> 0 (limbs go limp). Re-enter "
              "balance-stand from the app.")
        try:
            last = targets[min(stop, len(targets)) - 1]
            n = max(1, int(0.8 * (1 / dt)))
            for k in range(n + 1):
                send(last, 1.0 - k / n); time.sleep(dt)
        except Exception:
            pass
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Replay a SIREN trace on a real G1 (dry run by default).")
    p.add_argument("--trace", required=True, help="…_trace.csv")
    p.add_argument("--meta", help="…_trace_meta.json (default: next to --trace)")
    p.add_argument("--iface", default="en7", help="network interface for DDS")
    m = p.add_mutually_exclusive_group()
    m.add_argument("--go", action="store_true",
                   help="ACTUALLY MOVE THE ROBOT. Without this nothing is sent")
    m.add_argument("--check", action="store_true",
                   help="connect and compare the robot's pose to the trace's "
                        "first row, but send nothing")
    p.add_argument("--through-contact", action="store_true",
                   help="replay past contact_step. The trace collides there; "
                        "only use this with the obstacle physically absent")
    p.add_argument("--no-waist", dest="include_waist", action="store_false",
                   help="do NOT drive motors 12-14 (leave the waist damped). "
                        "The waist IS driven by default because holding it at "
                        "zero puts the hand ~180 mm off the recorded path")
    p.set_defaults(include_waist=True)
    p.add_argument("--waist-kp", type=float, default=WAIST_KP,
                   help=f"position gain for motors 12-14 (default "
                        f"{WAIST_KP}); not validated on hardware")
    p.add_argument("--rate", type=float,
                   help="override the replay rate in Hz (default: from the "
                        "metadata, which is how it was recorded)")
    p.add_argument("--takeover", type=float, default=1.5,
                   help="seconds to ramp stiffness in at the current pose")
    p.add_argument("--approach", type=float, default=2.0,
                   help="seconds to move from the current pose to trace row 0")
    p.add_argument("--max-approach", type=float, default=0.5,
                   help="refuse to start if row 0 is further than this (rad) "
                        "from where the arm actually is")
    p.add_argument("--limit-margin", type=float, default=0.05,
                   help="treat RealMotorPosLimit as this much tighter (rad)")
    p.add_argument("--show", type=int, default=8,
                   help="rows to print in the dry run")
    p.add_argument("--dump", help="dry run: write the full command stream here")
    a = p.parse_args(argv)

    rows, meta = load(a.trace, a.meta)
    dt = 1.0 / a.rate if a.rate else meta["timing"]["dt_s"]
    contact = meta["timing"].get("contact_step", -1)
    stop = len(rows)
    if contact is not None and contact >= 0 and not a.through_contact:
        stop = max(1, contact)

    joints, targets, notes, viol = plan(rows, meta, a.include_waist,
                                        a.limit_margin)
    stats = motion_stats(joints, targets[:stop], dt)
    print()
    report(rows, meta, joints, targets, stats, notes, viol, dt, stop, a)

    if viol and (a.go or a.check):
        raise SystemExit("\n  refusing to continue: the trace leaves the "
                         "robot's position limits (see above).")

    # Warn in EVERY mode, not just when arming. The dry run is where this is
    # meant to be noticed.
    if contact is not None and contact >= 0 and a.through_contact:
        print(f"\n  !! --through-contact: the replay INCLUDES the collision at "
              f"step {contact}.\n     This trajectory was searched for because "
              "it collides. Only run it on\n     hardware with the obstacle "
              "physically absent.")

    if a.go or a.check:
        return live(rows, meta, joints, targets, stop, dt, a)
    dry_run(rows, joints, targets, stop, a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
