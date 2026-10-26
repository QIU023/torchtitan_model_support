
### dp1 (reference dp1)

| cell | step 1 loss / grad norm | step 3 loss / grad norm | step 10 loss / grad norm | step-1 loss vs ref | loss parts at | grad norm parts at | step-1 gradients vs ref |
|---|---|---|---|---|---|---|---|
| dp1 | 8.0106010 / 1.6483026 | 6.03053 / 2.43840 | 3.41421 / 1.38656 | bitwise | never (20 steps) | never (20 steps) |  |
| dp1_again | 8.0106010 / 1.6483026 | 6.03053 / 2.43840 | 3.41421 / 1.38656 | bitwise | never (20 steps) | never (20 steps) | 159/159 bitwise, worst rel L2 0.00e+00 (tok_embeddings.weight), sign flips 0 |
| pp2 | 8.0106010 / 1.6483032 | 6.03037 / 2.43835 | 3.40580 / 1.33038 | bitwise | 2 | 1 |  |
| pp2_vpp2 | 8.0106010 / 1.6483032 | 6.03074 / 2.43825 | 3.41060 / 1.39249 | bitwise | 2 | 1 |  |
| pp4 | 8.0106010 / 1.6483032 | 6.03072 / 2.43831 | 3.41524 / 1.39797 | bitwise | 2 | 1 |  |
| pp8 | failed or short (0 steps): no error line | | | | | | |
| tp2_ep2 | 8.0106754 / 1.6482582 | 6.02941 / 2.43747 | 3.40727 / 1.32082 | +7.44e-05 | 1 | 1 | 43/159 bitwise, worst rel L2 2.70e-02 (layers.N.feed_forward.router.gate.weight), sign flips 35424 |
| tp4_ep4 | 8.0107775 / 1.6482801 | 6.02699 / 2.43622 | 3.40852 / 1.29376 | +1.76e-04 | 1 | 1 | 43/159 bitwise, worst rel L2 4.36e-02 (layers.N.feed_forward.router.gate.weight), sign flips 36484 |
| tp2_ep2_pp2 | 8.0106745 / 1.6482594 | 6.02954 / 2.43760 | 3.40730 / 1.31076 | +7.34e-05 | 1 | 1 |  |
| tp2_ep2_pp4 | failed or short (0 steps): no error line | | | | | | |
| tp4_ep4_pp2 | failed or short (0 steps): no error line | | | | | | |

### dp2 (reference fsdp2)

| cell | step 1 loss / grad norm | step 3 loss / grad norm | step 10 loss / grad norm | step-1 loss vs ref | loss parts at | grad norm parts at | step-1 gradients vs ref |
|---|---|---|---|---|---|---|---|
| fsdp2 | 8.0225201 / 1.7601706 | 6.00450 / 2.45491 | 3.54839 / 1.06422 | bitwise | never (20 steps) | never (20 steps) |  |
| fsdp2_again | 8.0225201 / 1.7601706 | 6.00450 / 2.45491 | 3.54839 / 1.06422 | bitwise | never (20 steps) | never (20 steps) | 159/159 bitwise, worst rel L2 0.00e+00 (tok_embeddings.weight), sign flips 0 |
| ep2_fsdp2 | 8.0225201 / 1.7601700 | 6.00409 / 2.45480 | 3.54760 / 1.09437 | bitwise | 2 | 1 | 145/159 bitwise, worst rel L2 1.15e-03 (layers.N.feed_forward.routed_experts.w13.weight), sign flips 856 |
| fsdp2_pp2 | 8.0225201 / 1.7601718 | 6.00413 / 2.45474 | 3.54872 / 1.07555 | bitwise | 2 | 1 |  |
| ep2_fsdp2_pp2 | 8.0225201 / 1.7601712 | 6.00439 / 2.45501 | 3.54732 / 1.06800 | bitwise | 2 | 1 |  |
| fsdp2_pp4 | failed or short (0 steps): no error line | | | | | | |
| ep2_fsdp2_pp4 | failed or short (0 steps): no error line | | | | | | |
| fsdp2_tp2_ep2 | 8.0225487 / 1.7601227 | 6.00204 / 2.45344 | 3.54846 / 1.09398 | +2.86e-05 | 1 | 1 | 43/159 bitwise, worst rel L2 2.80e-02 (layers.N.feed_forward.router.gate.weight), sign flips 34507 |
| fsdp2_tp2_ep4 | 8.0225487 / 1.7601236 | 6.00202 / 2.45348 | 3.54553 / 1.07082 | +2.86e-05 | 1 | 1 | 43/159 bitwise, worst rel L2 2.80e-02 (layers.N.feed_forward.router.gate.weight), sign flips 34447 |
| fsdp2_tp4_ep4 | failed or short (0 steps): no error line | | | | | | |
| fsdp2_tp4_ep8 | failed or short (0 steps): no error line | | | | | | |
| ep2_fsdp2_tp2_pp2 | failed or short (0 steps): no error line | | | | | | |
| ep4_fsdp2_tp2_pp2 | failed or short (0 steps): no error line | | | | | | |

### dp4 (reference fsdp4)

| cell | step 1 loss / grad norm | step 3 loss / grad norm | step 10 loss / grad norm | step-1 loss vs ref | loss parts at | grad norm parts at | step-1 gradients vs ref |
|---|---|---|---|---|---|---|---|
| fsdp4 | 8.0326843 / 1.7898487 | 6.30767 / 2.11076 | 3.55650 / 1.06354 | bitwise | never (20 steps) | never (20 steps) |  |
| fsdp4_again | 8.0326843 / 1.7898487 | 6.30767 / 2.11076 | 3.55650 / 1.06354 | bitwise | never (20 steps) | never (20 steps) | 159/159 bitwise, worst rel L2 0.00e+00 (tok_embeddings.weight), sign flips 0 |
| hsdp2x2 | 8.0326843 / 1.7898487 | 6.30767 / 2.11076 | 3.55937 / 1.07349 | bitwise | 5 | 4 | 126/159 bitwise, worst rel L2 5.22e-08 (layers.N.feed_forward.router.gate.weight), sign flips 0 |
| ep2_hsdp2x2 | 8.0326843 / 1.7898482 | 6.30753 / 2.11061 | 3.55765 / 1.07335 | bitwise | 2 | 1 | 124/159 bitwise, worst rel L2 1.17e-03 (layers.N.feed_forward.routed_experts.w13.weight), sign flips 835 |
| ep4_fsdp4 | 8.0326843 / 1.7898475 | 6.30766 / 2.11070 | 3.55757 / 1.07523 | bitwise | 2 | 1 | 145/159 bitwise, worst rel L2 1.41e-03 (layers.N.feed_forward.routed_experts.w13.weight), sign flips 376 |
| fsdp4_pp2 | failed or short (0 steps): no error line | | | | | | |
| ep4_fsdp4_pp2 | failed or short (0 steps): no error line | | | | | | |
| hsdp2x2_pp2 | failed or short (0 steps): no error line | | | | | | |
| fsdp4_tp2_ep2 | failed or short (0 steps): no error line | | | | | | |
| hsdp2x2_tp2_ep2 | failed or short (0 steps): no error line | | | | | | |

### dp8 (reference fsdp8)

| cell | step 1 loss / grad norm | step 3 loss / grad norm | step 10 loss / grad norm | step-1 loss vs ref | loss parts at | grad norm parts at | step-1 gradients vs ref |
|---|---|---|---|---|---|---|---|
| fsdp8 | failed or short (0 steps): no error line | | | | | | |
| fsdp8_again | failed or short (0 steps): no error line | | | | | | |
| ep8_fsdp8 | failed or short (0 steps): no error line | | | | | | |
| hsdp2x4 | failed or short (0 steps): no error line | | | | | | |
| hsdp4x2 | failed or short (0 steps): no error line | | | | | | |
| ep4_hsdp2x4 | failed or short (0 steps): no error line | | | | | | |

Full-precision step 1 per cell:
  dp1: loss 8.010601043701172 grad norm 1.6483025550842285
  dp1_again: loss 8.010601043701172 grad norm 1.6483025550842285
  pp2: loss 8.010601043701172 grad norm 1.6483031511306763
  pp2_vpp2: loss 8.010601043701172 grad norm 1.6483031511306763
  pp4: loss 8.010601043701172 grad norm 1.6483031511306763
  tp2_ep2: loss 8.010675430297852 grad norm 1.6482582092285156
  tp4_ep4: loss 8.010777473449707 grad norm 1.648280143737793
  tp2_ep2_pp2: loss 8.010674476623535 grad norm 1.6482594013214111
  fsdp2: loss 8.022520065307617 grad norm 1.760170578956604
  fsdp2_again: loss 8.022520065307617 grad norm 1.760170578956604
  ep2_fsdp2: loss 8.022520065307617 grad norm 1.7601699829101562
  fsdp2_pp2: loss 8.022520065307617 grad norm 1.7601717710494995
  ep2_fsdp2_pp2: loss 8.022520065307617 grad norm 1.7601711750030518
  fsdp2_tp2_ep2: loss 8.02254867553711 grad norm 1.7601226568222046
  fsdp2_tp2_ep4: loss 8.02254867553711 grad norm 1.760123610496521
  fsdp4: loss 8.032684326171875 grad norm 1.7898486852645874
  fsdp4_again: loss 8.032684326171875 grad norm 1.7898486852645874
  hsdp2x2: loss 8.032684326171875 grad norm 1.7898486852645874
  ep2_hsdp2x2: loss 8.032684326171875 grad norm 1.7898482084274292
  ep4_fsdp4: loss 8.032684326171875 grad norm 1.789847493171692
