
dp4, reference fsdp4:

| cell | loss, step 1 | step 10 | step 50 | step 100 | grad norm, step 1 | step 10 | step 50 | step 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fsdp4 | `8.032684` | `3.497031` | `3.007276` | `2.682869` | `1.7898` | `1.0201` | `0.6219` | `0.4218` |
| fsdp4, again | `8.032684`<br>identical | `3.497031`<br>identical | `3.007276`<br>identical | `2.682869`<br>identical | `1.7898`<br>identical | `1.0201`<br>identical | `0.6219`<br>identical | `0.4218`<br>identical |
| HSDP 2 x 2 | `8.032684`<br>identical | `3.494834`<br>-0.06% | `3.005924`<br>-0.04% | `2.682940`<br>+0.00% | `1.7898`<br>identical | `1.0071`<br>-1.27% | `0.6005`<br>-3.43% | `0.4254`<br>+0.86% |
