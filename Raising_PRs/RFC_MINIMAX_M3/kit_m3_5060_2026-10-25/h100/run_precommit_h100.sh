#!/bin/bash
# pre-commit on every file the baseline changes, in throwaway worktrees (Pyrefly rewrites suppressions repo-wide), and
# Pyrefly's error list on main for comparison.
. "$(dirname "$0")/env_h100.sh"
O=$R/precommit; mkdir -p "${O:?}"; P=$O/progress.txt; note() { echo "$(date +%H:%M:%S) $*" >> $P; }
. $V/bin/activate
MAIN=3ed5aa976fdf38e05fd50885ab192ea66ed6c5b7
for wt in lint_base lint_main; do git -C $B/tt worktree remove --force "${B:?}/$wt" 2>/dev/null; done
git -C $B/tt worktree add --detach $B/lint_base 9a5ab9ebe > /dev/null 2>&1
git -C $B/tt worktree add --detach $B/lint_main $MAIN > /dev/null 2>&1
files=$(git -C $B/lint_base diff --name-only --diff-filter=AM $MAIN HEAD)
( cd $B/lint_base && timeout 3000 pre-commit run --files $files > $O/baseline.log 2>&1 < /dev/null )
note "baseline rc=$? files=$(echo $files | wc -w) failed_hooks=[$(grep -E '\.\.\.\.Failed' $O/baseline.log | sed 's/\.\.*Failed//' | tr '\n' ',')]"
( cd $B/lint_main && timeout 3000 pre-commit run pyrefly-check --files torchtitan/models/common/norm.py > $O/main_pyrefly.log 2>&1 < /dev/null )
for t in baseline main_pyrefly; do grep -E "^ ?ERROR" $O/$t.log | sed -E 's/:[0-9]+:[0-9]+(-[0-9]+)?:/:/' | sort > $O/$t.err; done
note "pyrefly baseline [$(grep -E 'INFO .*errors' $O/baseline.log | head -1)] main [$(grep -E 'INFO .*errors' $O/main_pyrefly.log | head -1)] only_on_baseline=$(comm -13 $O/main_pyrefly.err $O/baseline.err | wc -l) only_on_main=$(comm -23 $O/main_pyrefly.err $O/baseline.err | wc -l)"
for wt in lint_base lint_main; do git -C $B/tt worktree remove --force "${B:?}/$wt"; done
note PRECOMMIT_DONE
