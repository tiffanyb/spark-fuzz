"""Launch a long experiment detached from this shell's process group.

Background jobs started from the agent shell die when the session tears down --
this driver has been killed twice that way mid-round, losing the in-flight work
each time. macOS has no setsid(1), so do it directly: os.setsid() puts the child
in a new session with no controlling terminal, so a process-group signal aimed at
the shell cannot reach it. The run is resumable anyway (coverage-driven), but not
being killed is better than resuming.

    python detach.py ./drive_goal.sh /tmp/goal6.log
"""
import os
import subprocess
import sys

cmd, log = sys.argv[1], sys.argv[2]
if os.fork():
    sys.exit(0)                      # parent returns to the shell immediately
os.setsid()                          # child leads a new session
with open(log, "ab", buffering=0) as fh:
    p = subprocess.Popen(["bash", cmd], stdout=fh, stderr=fh,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"detached pid {p.pid}", file=sys.stderr)
