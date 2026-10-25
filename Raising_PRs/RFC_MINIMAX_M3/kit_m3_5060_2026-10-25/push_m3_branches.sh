#!/bin/bash
# Push the 10-25 amend/rebase of the two M3 fork branches (approved by the user on 10-25): the old heads go to backup
# branches first, then each branch moves with --force-with-lease on its old head, all in one && chain.
set -o pipefail
cd /workspace/wt_m3 || exit 1
OLD_BASE=$(git rev-parse de7806dae) OLD_5D=$(git rev-parse 798f8817f)
NEW_BASE=$(git rev-parse 9a5ab9ebe) NEW_5D=$(git rev-parse 89fc094d1)
echo "remote before:"; git ls-remote origin refs/heads/minimax_m3 refs/heads/minimax_m3_5d 'refs/heads/backup/minimax_m3*'
[ "$(git ls-remote origin refs/heads/minimax_m3 | cut -f1)" = "$OLD_BASE" ] \
  && [ "$(git ls-remote origin refs/heads/minimax_m3_5d | cut -f1)" = "$OLD_5D" ] \
  && [ "$(git log --format=%B "$(git merge-base $NEW_BASE 3ed5aa976)".."$NEW_5D" | grep -c -i -E '^(co-authored-by|claude-session):')" = 0 ] \
  && git push origin "$OLD_BASE":refs/heads/backup/minimax_m3_pre_20261025 \
  && git push origin "$OLD_5D":refs/heads/backup/minimax_m3_5d_pre_20261025 \
  && git push --force-with-lease=refs/heads/minimax_m3:"$OLD_BASE" origin "$NEW_BASE":refs/heads/minimax_m3 \
  && git push --force-with-lease=refs/heads/minimax_m3_5d:"$OLD_5D" origin "$NEW_5D":refs/heads/minimax_m3_5d \
  && echo PUSH_CHAIN_OK
echo "remote after:"; git ls-remote origin refs/heads/minimax_m3 refs/heads/minimax_m3_5d 'refs/heads/backup/minimax_m3*'
echo "expected: minimax_m3 $NEW_BASE, minimax_m3_5d $NEW_5D, backups $OLD_BASE / $OLD_5D"
