# SIREN

**S**afe-control **I**nadequacy via **R**eachable-goal **E**xploitation **N**exus

SIREN studies a failure mode of reactive safety filters for robot manipulators
and humanoids: a single **admissible, reachable** intermediate goal `G1'`,
inserted before a legitimate goal `G1`, can trap the filter so the robot either
collides on the way back to `G1` or is denied the task altogether — even though
the same scene is safe without the inserted goal.

## Built on SPARK

SIREN is built on **SPARK v1** (https://github.com/intelligent-control-lab/spark).
For ease of reproducibility, we include the SPARK code at the corresponding
version in this repository.

## Installation

Follow the installation instructions in SPARK's own README:
https://github.com/intelligent-control-lab/spark/blob/main/README.md

## Running the experiments

Each experiment is self-contained and documented in its own directory. Run
everything from this repository root.

- **RQ1 — constructed goal-insertion attacks** — `fuzz/siren/experiment/rq1/`
  Constructs static-obstacle insertion attacks against SPARK's safety filters at
  the stock obstacle radius, for all value-based filters, and renders each one.
  See `fuzz/siren/experiment/rq1/README.md` (entry point: `run.sh`).

- **RQ3 — fuzzing over re-validated targets** — `fuzz/siren/experiment/rq3/`
  Runs SIREN's search over the re-validated insertion targets to measure how
  well its fuzzers rediscover ground-truth attacks.
  See `fuzz/siren/experiment/rq3/README.md` (entry point: `run_fuzz.sh`).

## Experiment 2

We are still on a debate on releasing the Experiment 2 scripts.
That experiment runs against
real humanoids, and the code could cause damage to the robots, the deployment
environment, and people around them.
Therefore, we are withholding the scripts until we finish the coordinated disclosure process.
