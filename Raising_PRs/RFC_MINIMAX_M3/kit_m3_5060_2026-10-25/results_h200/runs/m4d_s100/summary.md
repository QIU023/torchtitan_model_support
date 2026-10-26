
### dp1 (reference dp1)

| cell | step 1 loss / grad norm | step 10 loss / grad norm | step 50 loss / grad norm | step 100 loss / grad norm | step-1 loss vs ref | loss parts at | grad norm parts at | step-1 gradients vs ref |
|---|---|---|---|---|---|---|---|---|
| dp1 | 8.0106010 / 1.6483026 | 3.35214 / 1.35196 | 2.75800 / 0.64411 | 2.74373 / 0.69042 | bitwise | never (100 steps) | never (100 steps) |  |
| dp1_again | 8.0106010 / 1.6483026 | 3.35214 / 1.35196 | 2.75800 / 0.64411 | 2.74373 / 0.69042 | bitwise | never (100 steps) | never (100 steps) | 159/159 bitwise, worst rel L2 0.00e+00 (tok_embeddings.weight), sign flips 0 |
| pp2 | 8.0106010 / 1.6483032 | 3.34649 / 1.25961 | 2.75542 / 0.66065 | 2.74520 / 0.73503 | bitwise | 2 | 1 |  |
| pp2_vpp2 | 8.0106010 / 1.6483032 | 3.34480 / 1.30402 | 2.75076 / 0.60321 | 2.74101 / 0.69606 | bitwise | 2 | 1 |  |
| pp4 | 8.0106010 / 1.6483032 | 3.36363 / 1.43478 | 2.75552 / 0.60607 | 2.74325 / 0.69687 | bitwise | 2 | 1 |  |
| pp8 | failed or short (0 steps): no error line | | | | | | | |
| tp2_ep2 | 8.0106754 / 1.6482582 | 3.34878 / 1.30848 | 2.76201 / 0.72163 | 2.73884 / 0.69071 | +7.44e-05 | 1 | 1 | 43/159 bitwise, worst rel L2 2.70e-02 (layers.N.feed_forward.router.gate.weight), sign flips 35424 |
| tp4_ep4 | 8.0107775 / 1.6482801 | 3.34437 / 1.22297 | 2.76055 / 0.70874 | 2.74254 / 0.68878 | +1.76e-04 | 1 | 1 | 43/159 bitwise, worst rel L2 4.36e-02 (layers.N.feed_forward.router.gate.weight), sign flips 36484 |
| tp2_ep2_pp2 | 8.0106745 / 1.6482594 | 3.34493 / 1.26390 | 2.75685 / 0.58544 | 2.73933 / 0.69959 | +7.34e-05 | 1 | 1 |  |
| tp2_ep2_pp4 | failed or short (0 steps): no error line | | | | | | | |
| tp4_ep4_pp2 | failed or short (0 steps): no error line | | | | | | | |

### dp2 (reference fsdp2)

| cell | step 1 loss / grad norm | step 10 loss / grad norm | step 50 loss / grad norm | step 100 loss / grad norm | step-1 loss vs ref | loss parts at | grad norm parts at | step-1 gradients vs ref |
|---|---|---|---|---|---|---|---|---|
| fsdp2 | 8.0225201 / 1.7601706 | 3.48615 / 1.07276 | 2.71590 / 0.53572 | 2.68184 / 0.40761 | bitwise | never (100 steps) | never (100 steps) |  |
| fsdp2_again | 8.0225201 / 1.7601706 | 3.48615 / 1.07276 | 2.71590 / 0.53572 | 2.68184 / 0.40761 | bitwise | never (100 steps) | never (100 steps) | 159/159 bitwise, worst rel L2 0.00e+00 (tok_embeddings.weight), sign flips 0 |
| ep2_fsdp2 | 8.0225201 / 1.7601700 | 3.49008 / 1.14021 | 2.71963 / 0.62217 | 2.67981 / 0.38125 | bitwise | 2 | 1 | 145/159 bitwise, worst rel L2 1.15e-03 (layers.N.feed_forward.routed_experts.w13.weight), sign flips 856 |
| fsdp2_pp2 | 8.0225201 / 1.7601718 | 3.49376 / 1.13764 | 2.72021 / 0.63694 | 2.68360 / 0.38515 | bitwise | 2 | 1 |  |
| ep2_fsdp2_pp2 | 8.0225201 / 1.7601712 | 3.49769 / 1.20646 | 2.72217 / 0.61470 | 2.68063 / 0.36583 | bitwise | 2 | 1 |  |
| fsdp2_pp4 | failed or short (0 steps): no error line | | | | | | | |
| ep2_fsdp2_pp4 | failed or short (0 steps): no error line | | | | | | | |
| fsdp2_tp2_ep2 | 8.0225487 / 1.7601227 | 3.48466 / 1.00671 | 2.71854 / 0.64043 | 2.68113 / 0.38301 | +2.86e-05 | 1 | 1 | 43/159 bitwise, worst rel L2 2.80e-02 (layers.N.feed_forward.router.gate.weight), sign flips 34507 |
| fsdp2_tp2_ep4 | 8.0225487 / 1.7601236 | 3.49013 / 1.00574 | 2.72125 / 0.62869 | 2.68086 / 0.38682 | +2.86e-05 | 1 | 1 | 43/159 bitwise, worst rel L2 2.80e-02 (layers.N.feed_forward.router.gate.weight), sign flips 34447 |
| fsdp2_tp4_ep4 | failed or short (0 steps): no error line | | | | | | | |
| fsdp2_tp4_ep8 | failed or short (0 steps): no error line | | | | | | | |
| ep2_fsdp2_tp2_pp2 | failed or short (0 steps): no error line | | | | | | | |
| ep4_fsdp2_tp2_pp2 | failed or short (0 steps): no error line | | | | | | | |

### dp4 (reference fsdp4)

| cell | step 1 loss / grad norm | step 10 loss / grad norm | step 50 loss / grad norm | step 100 loss / grad norm | step-1 loss vs ref | loss parts at | grad norm parts at | step-1 gradients vs ref |
|---|---|---|---|---|---|---|---|---|
| fsdp4 | 8.0326843 / 1.7898487 | 3.49703 / 1.02008 | 3.00728 / 0.62187 | 2.68297 / 0.42384 | bitwise | never (100 steps) | never (100 steps) |  |
| fsdp4_again | 8.0326843 / 1.7898487 | 3.49703 / 1.02008 | 3.00728 / 0.62187 | 2.68297 / 0.42384 | bitwise | never (100 steps) | never (100 steps) | 159/159 bitwise, worst rel L2 0.00e+00 (tok_embeddings.weight), sign flips 0 |
| ep4_fsdp4 | 8.0326843 / 1.7898475 | 3.49383 / 0.98151 | 3.00472 / 0.59333 | 2.68426 / 0.41316 | bitwise | 2 | 1 | 145/159 bitwise, worst rel L2 1.41e-03 (layers.N.feed_forward.routed_experts.w13.weight), sign flips 376 |
| fsdp4_pp2 | failed or short (0 steps): no error line | | | | | | | |
| ep4_fsdp4_pp2 | failed or short (0 steps): no error line | | | | | | | |
| fsdp4_tp2_ep2 | failed or short (0 steps): no error line | | | | | | | |

### dp8 (reference fsdp8)

| cell | step 1 loss / grad norm | step 10 loss / grad norm | step 50 loss / grad norm | step 100 loss / grad norm | step-1 loss vs ref | loss parts at | grad norm parts at | step-1 gradients vs ref |
|---|---|---|---|---|---|---|---|---|
| fsdp8 | failed or short (0 steps): no error line | | | | | | | |
| fsdp8_again | failed or short (0 steps): no error line | | | | | | | |
| ep8_fsdp8 | failed or short (0 steps): no error line | | | | | | | |

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
  ep4_fsdp4: loss 8.032684326171875 grad norm 1.789847493171692
