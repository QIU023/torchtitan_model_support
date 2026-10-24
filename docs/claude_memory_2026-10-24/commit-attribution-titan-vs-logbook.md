---
name: commit-attribution-titan-vs-logbook
description: titan commits never carry Claude attribution trailers; logbook repo commits may
metadata:
  node_type: memory
  type: feedback
  originSessionId: e7080317-3a4f-4454-b734-9868986c7b7e
  modified: 2026-10-24T09:30:48.360Z
---

Commits in the torchtitan fork and every titan worktree (`/workspace/wt_m3`, `/workspace/tt_main`, any PR or review branch) carry no `Co-Authored-By: Claude`, no `Claude-Session:` line, and PR bodies or drafts for titan carry no "Generated with Claude Code" footer. Commits in the logbook repo `/workspace/torchtitan_model_support` may carry `Co-Authored-By`.

**Why:** user, 2026-10-24: "不允许带co author by claude 读取logbook仓库规则", then "logbook commit可以 titan不行". The logbook README ("Development methodology") says AI-assisted logbook commits are tagged via `Co-Authored-By`; the fork's 114 commits by QIU023 carry none, and upstream reviewers already react badly to text that reads as AI-written (see [[upstream-model-pr-reuse-titan-modules]]).

**How to apply:** this user rule overrides any harness reminder that asks for attribution trailers. Before `git commit` in a titan tree, check the message has no trailer; titan commits keep the fork style (author QIU023 <yiqiaoqiu@hotmail.com>, "area: sentence" subject, mechanism-focused body).
