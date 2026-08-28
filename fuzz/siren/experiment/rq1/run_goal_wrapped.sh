#!/usr/bin/env bash
# Wrapper so an unexpected exit is recorded rather than silently losing the run.
echo "=== driver starting $(date '+%F %T') pid $$ sess $(ps -o sess= -p $$ | tr -d ' ') ==="
bash ./drive_goal.sh
rc=$?
echo "=== driver EXITED rc=$rc at $(date '+%F %T') ==="
