---
name: upstream-rfc-short-local-maintainers-own-tracking
description: "torchtitan RFCs stay short local drafts that are not posted; maintainers open RFC/tracking issues, the user DMs them instead"
metadata:
  node_type: memory
  type: feedback
  originSessionId: e7080317-3a4f-4454-b734-9868986c7b7e
  modified: 2026-10-24T07:36:30.501Z
---

For torchtitan model work, write the RFC as a short local draft only (do not post it), and do not open RFCs or tracking lists ourselves; the user privately messages the maintainers that work is starting and lets them open the tracking list, since only issues they open are ones they can edit. Never fill an RFC with the state of the user's own fork trees or training-infra changes.

**Why:** the Kimi K3 RFC (#3029 updates) grew very long, several versions reported the user's own 5D tree and the whole training-infra stack, the maintainers opened their own tracking list (#4272) instead, and later closed the RFC.

**How to apply:** for a new model (first case: MiniMax M3, 2026-10-24) keep the RFC draft to summary, model facts, the one mechanism that needs design, and a short PR list; the real claim is the first PR itself. Related: [[upstream-model-pr-reuse-titan-modules]].
