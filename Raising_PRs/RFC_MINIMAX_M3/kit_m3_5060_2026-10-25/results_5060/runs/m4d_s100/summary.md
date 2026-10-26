
### dp1 (reference dp1)

| cell | step 1 loss / grad norm | step 10 loss / grad norm | step 50 loss / grad norm | step 100 loss / grad norm | step-1 loss vs ref | loss parts at | grad norm parts at | step-1 gradients vs ref |
|---|---|---|---|---|---|---|---|---|
| dp1 | 8.0105877 / 1.6482835 | 3.33881 / 1.19643 | 2.75774 / 0.61914 | 2.74239 / 0.69133 | bitwise | never (100 steps) | never (100 steps) |  |
| dp1_again | 8.0105877 / 1.6482835 | 3.33881 / 1.19643 | 2.75774 / 0.61914 | 2.74239 / 0.69133 | bitwise | never (100 steps) | never (100 steps) | 159/159 bitwise, worst rel L2 0.00e+00 (tok_embeddings.weight), sign flips 0 |
| pp2 | 8.0105877 / 1.6482846 | 3.34189 / 1.20094 | 2.76236 / 0.70644 | 2.74515 / 0.69388 | bitwise | 2 | 1 |  |
| pp2_vpp2 | 8.0105877 / 1.6482843 | 3.34428 / 1.28479 | 2.75521 / 0.61500 | 2.74440 / 0.70033 | bitwise | 2 | 1 |  |
| pp4 | 8.0105877 / 1.6482847 | 3.34608 / 1.20865 | 2.75718 / 0.63659 | 2.74253 / 0.71892 | bitwise | 2 | 1 |  |
| pp8 | 8.0105877 / 1.6482847 | 3.34788 / 1.21047 | 2.75499 / 0.60524 | 2.73844 / 0.69043 | bitwise | 2 | 1 |  |
| tp2_ep2 | 8.0107040 / 1.6482679 | 3.34096 / 1.25181 | 2.75906 / 0.65101 | 2.74251 / 0.69766 | +1.16e-04 | 1 | 1 | 43/159 bitwise, worst rel L2 2.79e-02 (layers.N.feed_forward.router.gate.weight), sign flips 35340 |
| tp4_ep4 | 8.0107594 / 1.6482800 | 3.34005 / 1.22537 | 2.76017 / 0.61562 | 2.74233 / 0.69775 | +1.72e-04 | 1 | 1 | 43/159 bitwise, worst rel L2 3.93e-02 (layers.N.feed_forward.router.gate.weight), sign flips 36542 |
| tp2_ep2_pp2 | 8.0107031 / 1.6482688 | 3.35303 / 1.36941 | 2.76028 / 0.68370 | 2.73997 / 0.70613 | +1.15e-04 | 1 | 1 |  |
| tp2_ep2_pp4 | 8.0107031 / 1.6482689 | 3.34207 / 1.27007 | 2.75471 / 0.59599 | 2.74219 / 0.71351 | +1.15e-04 | 1 | 1 |  |
| tp4_ep4_pp2 | 8.0107594 / 1.6482815 | 3.35668 / 1.36765 | 2.75962 / 0.62388 | 2.74467 / 0.71002 | +1.72e-04 | 1 | 1 |  |

### dp2 (reference fsdp2)

| cell | step 1 loss / grad norm | step 10 loss / grad norm | step 50 loss / grad norm | step 100 loss / grad norm | step-1 loss vs ref | loss parts at | grad norm parts at | step-1 gradients vs ref |
|---|---|---|---|---|---|---|---|---|
| fsdp2 | 8.0225143 / 1.7601597 | 3.48779 / 1.03293 | 2.71606 / 0.53836 | 2.68012 / 0.39385 | bitwise | never (100 steps) | never (100 steps) |  |
| fsdp2_again | 8.0225143 / 1.7601597 | 3.48779 / 1.03293 | 2.71606 / 0.53836 | 2.68012 / 0.39385 | bitwise | never (100 steps) | never (100 steps) | 159/159 bitwise, worst rel L2 0.00e+00 (tok_embeddings.weight), sign flips 0 |
| ep2_fsdp2 | 8.0225143 / 1.7601593 | 3.49835 / 1.19216 | 2.71839 / 0.66029 | 2.68078 / 0.37050 | bitwise | 2 | 1 | 145/159 bitwise, worst rel L2 1.15e-03 (layers.N.feed_forward.routed_experts.w13.weight), sign flips 852 |
| fsdp2_pp2 | 8.0225143 / 1.7601619 | 3.49058 / 1.11404 | 2.71634 / 0.57623 | 2.68261 / 0.38067 | bitwise | 2 | 1 |  |
| ep2_fsdp2_pp2 | 8.0225143 / 1.7601616 | 3.48691 / 0.99965 | 2.71886 / 0.60859 | 2.68282 / 0.37573 | bitwise | 2 | 1 |  |
| fsdp2_pp4 | 8.0225143 / 1.7601620 | 3.48765 / 1.06082 | 2.71846 / 0.62908 | 2.68246 / 0.38237 | bitwise | 2 | 1 |  |
| ep2_fsdp2_pp4 | 8.0225143 / 1.7601616 | 3.48755 / 1.03987 | 2.71854 / 0.60551 | 2.68063 / 0.38958 | bitwise | 2 | 1 |  |
| fsdp2_tp2_ep2 | 8.0225945 / 1.7601259 | 3.49832 / 1.21322 | 2.71754 / 0.59872 | 2.68288 / 0.36504 | +8.01e-05 | 1 | 1 | 43/159 bitwise, worst rel L2 2.54e-02 (layers.N.feed_forward.router.gate.weight), sign flips 34748 |
| fsdp2_tp2_ep4 | 8.0225945 / 1.7601260 | 3.48651 / 1.03363 | 2.71882 / 0.63241 | 2.68013 / 0.39637 | +8.01e-05 | 1 | 1 | 43/159 bitwise, worst rel L2 2.54e-02 (layers.N.feed_forward.router.gate.weight), sign flips 34614 |
| fsdp2_tp4_ep4 | 8.0225611 / 1.7601511 | 3.49308 / 1.17344 | 2.71489 / 0.55823 | 2.68189 / 0.37892 | +4.67e-05 | 1 | 1 | 43/159 bitwise, worst rel L2 2.98e-02 (layers.N.feed_forward.router.gate.weight), sign flips 38800 |
| fsdp2_tp4_ep8 | 8.0225611 / 1.7601517 | 3.49393 / 1.15529 | 2.72029 / 0.62042 | 2.68221 / 0.38294 | +4.67e-05 | 1 | 1 | 43/159 bitwise, worst rel L2 2.98e-02 (layers.N.feed_forward.router.gate.weight), sign flips 38712 |
| ep2_fsdp2_tp2_pp2 | 8.0225945 / 1.7601264 | 3.49049 / 1.04341 | 2.72260 / 0.63628 | 2.68132 / 0.37942 | +8.01e-05 | 1 | 1 |  |
| ep4_fsdp2_tp2_pp2 | 8.0225945 / 1.7601264 | 3.48787 / 1.06360 | 2.72294 / 0.62879 | 2.68125 / 0.39856 | +8.01e-05 | 1 | 1 |  |

### dp4 (reference fsdp4)

| cell | step 1 loss / grad norm | step 10 loss / grad norm | step 50 loss / grad norm | step 100 loss / grad norm | step-1 loss vs ref | loss parts at | grad norm parts at | step-1 gradients vs ref |
|---|---|---|---|---|---|---|---|---|
| fsdp4 | 8.0326929 / 1.7898140 | 3.49115 / 0.98930 | 3.01166 / 0.66384 | 2.68458 / 0.43379 | bitwise | never (100 steps) | never (100 steps) |  |
| fsdp4_again | 8.0326929 / 1.7898140 | 3.49115 / 0.98930 | 3.01166 / 0.66384 | 2.68458 / 0.43379 | bitwise | never (100 steps) | never (100 steps) | 159/159 bitwise, worst rel L2 0.00e+00 (tok_embeddings.weight), sign flips 0 |
| ep4_fsdp4 | 8.0326929 / 1.7898129 | 3.49280 / 0.99250 | 3.00022 / 0.52476 | 2.68318 / 0.42090 | bitwise | 2 | 1 | 145/159 bitwise, worst rel L2 1.62e-03 (layers.N.feed_forward.routed_experts.w13.weight), sign flips 388 |
| fsdp4_pp2 | 8.0326929 / 1.7898147 | 3.49438 / 1.01902 | 3.00809 / 0.63592 | 2.68380 / 0.43060 | bitwise | 2 | 1 |  |
| ep4_fsdp4_pp2 | 8.0326929 / 1.7898136 | 3.49484 / 0.99230 | 3.01097 / 0.62687 | 2.68161 / 0.42292 | bitwise | 2 | 1 |  |
| fsdp4_tp2_ep2 | 8.0327034 / 1.7897980 | 3.49691 / 1.00658 | 3.00386 / 0.56938 | 2.68257 / 0.41775 | +1.05e-05 | 1 | 1 | 43/159 bitwise, worst rel L2 3.58e-02 (layers.N.feed_forward.router.gate.weight), sign flips 34502 |

### dp8 (reference fsdp8)

| cell | step 1 loss / grad norm | step 10 loss / grad norm | step 50 loss / grad norm | step 100 loss / grad norm | step-1 loss vs ref | loss parts at | grad norm parts at | step-1 gradients vs ref |
|---|---|---|---|---|---|---|---|---|
| fsdp8 | 8.0218401 / 1.7600516 | 3.41540 / 1.19589 | 2.76523 / 0.55930 | 2.79922 / 0.40573 | bitwise | never (100 steps) | never (100 steps) |  |
| fsdp8_again | 8.0218401 / 1.7600516 | 3.41540 / 1.19589 | 2.76523 / 0.55930 | 2.79922 / 0.40573 | bitwise | never (100 steps) | never (100 steps) | 159/159 bitwise, worst rel L2 0.00e+00 (tok_embeddings.weight), sign flips 0 |
| ep8_fsdp8 | 8.0218401 / 1.7600508 | 3.42021 / 1.30918 | 2.76814 / 0.60603 | 2.80040 / 0.41674 | bitwise | 2 | 1 | 145/159 bitwise, worst rel L2 2.39e-03 (layers.N.feed_forward.routed_experts.w13.weight), sign flips 619 |

Full-precision step 1 per cell:
  dp1: loss 8.010587692260742 grad norm 1.6482834815979004
  dp1_again: loss 8.010587692260742 grad norm 1.6482834815979004
  pp2: loss 8.010587692260742 grad norm 1.6482845544815063
  pp2_vpp2: loss 8.010587692260742 grad norm 1.6482843160629272
  pp4: loss 8.010587692260742 grad norm 1.648284673690796
  pp8: loss 8.010587692260742 grad norm 1.648284673690796
  tp2_ep2: loss 8.010704040527344 grad norm 1.6482678651809692
  tp4_ep4: loss 8.010759353637695 grad norm 1.6482800245285034
  tp2_ep2_pp2: loss 8.010703086853027 grad norm 1.6482688188552856
  tp2_ep2_pp4: loss 8.010703086853027 grad norm 1.6482689380645752
  tp4_ep4_pp2: loss 8.010759353637695 grad norm 1.648281455039978
  fsdp2: loss 8.022514343261719 grad norm 1.7601597309112549
  fsdp2_again: loss 8.022514343261719 grad norm 1.7601597309112549
  ep2_fsdp2: loss 8.022514343261719 grad norm 1.7601592540740967
  fsdp2_pp2: loss 8.022514343261719 grad norm 1.7601618766784668
  ep2_fsdp2_pp2: loss 8.022514343261719 grad norm 1.7601616382598877
  fsdp2_pp4: loss 8.022514343261719 grad norm 1.7601619958877563
  ep2_fsdp2_pp4: loss 8.022514343261719 grad norm 1.7601616382598877
  fsdp2_tp2_ep2: loss 8.022594451904297 grad norm 1.7601258754730225
  fsdp2_tp2_ep4: loss 8.022594451904297 grad norm 1.760125994682312
  fsdp2_tp4_ep4: loss 8.022561073303223 grad norm 1.7601511478424072
  fsdp2_tp4_ep8: loss 8.022561073303223 grad norm 1.760151743888855
  ep2_fsdp2_tp2_pp2: loss 8.022594451904297 grad norm 1.7601263523101807
  ep4_fsdp2_tp2_pp2: loss 8.022594451904297 grad norm 1.7601263523101807
  fsdp4: loss 8.032692909240723 grad norm 1.7898139953613281
  fsdp4_again: loss 8.032692909240723 grad norm 1.7898139953613281
  ep4_fsdp4: loss 8.032692909240723 grad norm 1.7898129224777222
  fsdp4_pp2: loss 8.032692909240723 grad norm 1.7898147106170654
  ep4_fsdp4_pp2: loss 8.032692909240723 grad norm 1.7898136377334595
  fsdp4_tp2_ep2: loss 8.032703399658203 grad norm 1.7897980213165283
  fsdp8: loss 8.02184009552002 grad norm 1.7600516080856323
  fsdp8_again: loss 8.02184009552002 grad norm 1.7600516080856323
  ep8_fsdp8: loss 8.02184009552002 grad norm 1.7600507736206055
