
### dp1 (reference dp1)

| cell | step 1 loss / grad norm | step 3 | step 10 | step-1 loss vs ref | loss parts at | grad norm parts at | step-1 gradients vs ref |
|---|---|---|---|---|---|---|---|
| dp1 | 8.0105877 / 1.6482835 | 6.02989 / 2.43819 | 3.41077 / 1.31677 | bitwise | never (20 steps) | never (20 steps) |  |
| dp1_again | 8.0105877 / 1.6482835 | 6.02989 / 2.43819 | 3.41077 / 1.31677 | bitwise | never (20 steps) | never (20 steps) | 159/159 bitwise, worst rel L2 0.00e+00 (tok_embeddings.weight), sign flips 0 |
| pp2 | 8.0105877 / 1.6482846 | 6.03024 / 2.43841 | 3.40928 / 1.33348 | bitwise | 2 | 1 |  |
| pp2_vpp2 | 8.0105877 / 1.6482843 | 6.03005 / 2.43820 | 3.41515 / 1.39657 | bitwise | 2 | 1 |  |
| pp4 | 8.0105877 / 1.6482847 | 6.03028 / 2.43840 | 3.40782 / 1.30352 | bitwise | 2 | 1 |  |
| pp8 | 8.0105877 / 1.6482847 | 6.03029 / 2.43837 | 3.40876 / 1.29200 | bitwise | 2 | 1 |  |
| tp2_ep2 | 8.0107040 / 1.6482679 | 6.02935 / 2.43724 | 3.40901 / 1.29799 | +1.16e-04 | 1 | 1 | 43/159 bitwise, worst rel L2 2.79e-02 (layers.N.feed_forward.router.gate.weight), sign flips 35340 |
| tp4_ep4 | 8.0107594 / 1.6482800 | 6.02697 / 2.43628 | 3.41173 / 1.33967 | +1.72e-04 | 1 | 1 | 43/159 bitwise, worst rel L2 3.93e-02 (layers.N.feed_forward.router.gate.weight), sign flips 36542 |
| tp2_ep2_pp2 | 8.0107031 / 1.6482688 | 6.02786 / 2.43676 | 3.41163 / 1.36842 | +1.15e-04 | 1 | 1 |  |
| tp2_ep2_pp4 | 8.0107031 / 1.6482689 | 6.02794 / 2.43681 | 3.41030 / 1.35596 | +1.15e-04 | 1 | 1 |  |
| tp4_ep4_pp2 | 8.0107594 / 1.6482815 | 6.02715 / 2.43623 | 3.41477 / 1.31117 | +1.72e-04 | 1 | 1 |  |

### dp2 (reference fsdp2)

| cell | step 1 loss / grad norm | step 3 | step 10 | step-1 loss vs ref | loss parts at | grad norm parts at | step-1 gradients vs ref |
|---|---|---|---|---|---|---|---|
| fsdp2 | 8.0225143 / 1.7601597 | 6.00461 / 2.45486 | 3.54976 / 1.07344 | bitwise | never (20 steps) | never (20 steps) |  |
| fsdp2_again | 8.0225143 / 1.7601597 | 6.00461 / 2.45486 | 3.54976 / 1.07344 | bitwise | never (20 steps) | never (20 steps) | 159/159 bitwise, worst rel L2 0.00e+00 (tok_embeddings.weight), sign flips 0 |
| ep2_fsdp2 | 8.0225143 / 1.7601593 | 6.00417 / 2.45459 | 3.54964 / 1.06043 | bitwise | 2 | 1 | 145/159 bitwise, worst rel L2 1.15e-03 (layers.N.feed_forward.routed_experts.w13.weight), sign flips 852 |
| fsdp2_pp2 | 8.0225143 / 1.7601619 | 6.00435 / 2.45491 | 3.54790 / 1.07185 | bitwise | 2 | 1 |  |
| ep2_fsdp2_pp2 | 8.0225143 / 1.7601616 | 6.00423 / 2.45476 | 3.54740 / 1.06881 | bitwise | 2 | 1 |  |
| fsdp2_pp4 | 8.0225143 / 1.7601620 | 6.00431 / 2.45470 | 3.54814 / 1.08436 | bitwise | 2 | 1 |  |
| ep2_fsdp2_pp4 | 8.0225143 / 1.7601616 | 6.00444 / 2.45472 | 3.54811 / 1.06565 | bitwise | 2 | 1 |  |
| fsdp2_tp2_ep2 | 8.0225945 / 1.7601259 | 6.00036 / 2.45181 | 3.55206 / 1.12536 | +8.01e-05 | 1 | 1 | 43/159 bitwise, worst rel L2 2.54e-02 (layers.N.feed_forward.router.gate.weight), sign flips 34748 |
| fsdp2_tp2_ep4 | 8.0225945 / 1.7601260 | 6.00076 / 2.45190 | 3.54608 / 1.06823 | +8.01e-05 | 1 | 1 | 43/159 bitwise, worst rel L2 2.54e-02 (layers.N.feed_forward.router.gate.weight), sign flips 34614 |
| fsdp2_tp4_ep4 | 8.0225611 / 1.7601511 | 6.00174 / 2.45394 | 3.55399 / 1.17156 | +4.67e-05 | 1 | 1 | 43/159 bitwise, worst rel L2 2.98e-02 (layers.N.feed_forward.router.gate.weight), sign flips 38800 |
| fsdp2_tp4_ep8 | 8.0225611 / 1.7601517 | 6.00202 / 2.45363 | 3.55409 / 1.18777 | +4.67e-05 | 1 | 1 | 43/159 bitwise, worst rel L2 2.98e-02 (layers.N.feed_forward.router.gate.weight), sign flips 38712 |
| ep2_fsdp2_tp2_pp2 | 8.0225945 / 1.7601264 | 6.00030 / 2.45214 | 3.55417 / 1.15398 | +8.01e-05 | 1 | 1 |  |
| ep4_fsdp2_tp2_pp2 | 8.0225945 / 1.7601264 | 6.00052 / 2.45256 | 3.55284 / 1.16666 | +8.01e-05 | 1 | 1 |  |

### dp4 (reference fsdp4)

| cell | step 1 loss / grad norm | step 3 | step 10 | step-1 loss vs ref | loss parts at | grad norm parts at | step-1 gradients vs ref |
|---|---|---|---|---|---|---|---|
| fsdp4 | 8.0326929 / 1.7898140 | 6.30805 / 2.11084 | 3.56073 / 1.11180 | bitwise | never (20 steps) | never (20 steps) |  |
| fsdp4_again | 8.0326929 / 1.7898140 | 6.30805 / 2.11084 | 3.56073 / 1.11180 | bitwise | never (20 steps) | never (20 steps) | 159/159 bitwise, worst rel L2 0.00e+00 (tok_embeddings.weight), sign flips 0 |
| hsdp2x2 | 8.0326929 / 1.7898140 | 6.30805 / 2.11084 | 3.55743 / 1.08442 | bitwise | 4 | 3 | 117/159 bitwise, worst rel L2 5.27e-08 (layers.N.feed_forward.router.gate.weight), sign flips 0 |
| ep2_hsdp2x2 | 8.0326929 / 1.7898149 | 6.30813 / 2.11085 | 3.55683 / 1.08127 | bitwise | 2 | 1 | 117/159 bitwise, worst rel L2 1.17e-03 (layers.N.feed_forward.routed_experts.w13.weight), sign flips 828 |
| ep4_fsdp4 | 8.0326929 / 1.7898129 | 6.30790 / 2.11085 | 3.55631 / 1.06888 | bitwise | 2 | 1 | 145/159 bitwise, worst rel L2 1.62e-03 (layers.N.feed_forward.routed_experts.w13.weight), sign flips 388 |
| fsdp4_pp2 | 8.0326929 / 1.7898147 | 6.30781 / 2.11083 | 3.55729 / 1.07458 | bitwise | 2 | 1 |  |
| ep4_fsdp4_pp2 | 8.0326929 / 1.7898136 | 6.30777 / 2.11082 | 3.55618 / 1.07779 | bitwise | 2 | 1 |  |
| hsdp2x2_pp2 | 8.0326929 / 1.7898147 | 6.30780 / 2.11083 | 3.55879 / 1.09424 | bitwise | 2 | 1 |  |
| fsdp4_tp2_ep2 | 8.0327034 / 1.7897980 | 6.30973 / 2.11097 | 3.55946 / 1.08769 | +1.05e-05 | 1 | 1 | 43/159 bitwise, worst rel L2 3.58e-02 (layers.N.feed_forward.router.gate.weight), sign flips 34502 |
| hsdp2x2_tp2_ep2 | 8.0327034 / 1.7897980 | 6.30973 / 2.11097 | 3.55543 / 1.08740 | +1.05e-05 | 1 | 1 | 43/159 bitwise, worst rel L2 3.58e-02 (layers.N.feed_forward.router.gate.weight), sign flips 34502 |

### dp8 (reference fsdp8)

| cell | step 1 loss / grad norm | step 3 | step 10 | step-1 loss vs ref | loss parts at | grad norm parts at | step-1 gradients vs ref |
|---|---|---|---|---|---|---|---|
| fsdp8 | 8.0218401 / 1.7600516 | 6.09688 / 2.33329 | 3.48511 / 1.41856 | bitwise | never (20 steps) | never (20 steps) |  |
| fsdp8_again | 8.0218401 / 1.7600516 | 6.09688 / 2.33329 | 3.48511 / 1.41856 | bitwise | never (20 steps) | never (20 steps) | 159/159 bitwise, worst rel L2 0.00e+00 (tok_embeddings.weight), sign flips 0 |
| ep8_fsdp8 | 8.0218401 / 1.7600508 | 6.09659 / 2.33325 | 3.48368 / 1.39608 | bitwise | 2 | 1 | 145/159 bitwise, worst rel L2 2.39e-03 (layers.N.feed_forward.routed_experts.w13.weight), sign flips 619 |
| hsdp2x4 | 8.0218401 / 1.7600516 | 6.09688 / 2.33329 | 3.47841 / 1.28843 | bitwise | 4 | 3 | 118/159 bitwise, worst rel L2 6.45e-08 (layers.N.feed_forward.router.gate.weight), sign flips 0 |
| hsdp4x2 | 8.0218401 / 1.7600516 | 6.09688 / 2.33329 | 3.47953 / 1.30646 | bitwise | 4 | 3 | 117/159 bitwise, worst rel L2 6.44e-08 (layers.N.feed_forward.router.gate.weight), sign flips 0 |
| ep4_hsdp2x4 | 8.0218401 / 1.7600518 | 6.09639 / 2.33306 | 3.47833 / 1.29568 | bitwise | 2 | 1 | 118/159 bitwise, worst rel L2 1.63e-03 (layers.N.feed_forward.routed_experts.w13.weight), sign flips 394 |

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
  hsdp2x2: loss 8.032692909240723 grad norm 1.7898139953613281
  ep2_hsdp2x2: loss 8.032692909240723 grad norm 1.7898149490356445
  ep4_fsdp4: loss 8.032692909240723 grad norm 1.7898129224777222
  fsdp4_pp2: loss 8.032692909240723 grad norm 1.7898147106170654
  ep4_fsdp4_pp2: loss 8.032692909240723 grad norm 1.7898136377334595
  hsdp2x2_pp2: loss 8.032692909240723 grad norm 1.7898147106170654
  fsdp4_tp2_ep2: loss 8.032703399658203 grad norm 1.7897980213165283
  hsdp2x2_tp2_ep2: loss 8.032703399658203 grad norm 1.7897980213165283
  fsdp8: loss 8.02184009552002 grad norm 1.7600516080856323
  fsdp8_again: loss 8.02184009552002 grad norm 1.7600516080856323
  ep8_fsdp8: loss 8.02184009552002 grad norm 1.7600507736206055
  hsdp2x4: loss 8.02184009552002 grad norm 1.7600516080856323
  hsdp4x2: loss 8.02184009552002 grad norm 1.7600516080856323
  ep4_hsdp2x4: loss 8.02184009552002 grad norm 1.7600518465042114
