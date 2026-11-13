| k | backend | lang | train s (median) | vs Rust | assign s (median) | vs Rust | peak RSS MB | runs | = Rust |
|---|---|---|---|---|---|---|---|---|---|
| 64 | cpu-fused | rust | 0.587 | — | 0.038 | — | 612 | 5 | yes |
| 64 | cpu-fused | python | 0.535 | -8.7% | 0.030 | -19.1% | 649 | 5 | yes |
| 64 | cpu-fused | node | 0.624 | +6.3% | 0.040 | +5.1% | 666 | 5 | yes |
| 64 | cpu-fused | go | 0.540 | -8.0% | 0.035 | -6.5% | 618 | 5 | yes |
| 256 | cpu-fused | rust | 0.669 | — | 0.064 | — | 614 | 5 | yes |
| 256 | cpu-fused | python | 0.666 | -0.4% | 0.072 | +11.6% | 649 | 5 | yes |
| 256 | cpu-fused | node | 0.685 | +2.5% | 0.071 | +10.3% | 670 | 5 | yes |
| 256 | cpu-fused | go | 0.673 | +0.7% | 0.071 | +11.2% | 625 | 5 | yes |
| 1024 | cpu-fused | rust | 1.678 | — | 0.219 | — | 637 | 5 | yes |
| 1024 | cpu-fused | python | 1.728 | +3.0% | 0.220 | +0.8% | 672 | 5 | yes |
| 1024 | cpu-fused | node | 1.692 | +0.8% | 0.214 | -2.2% | 695 | 5 | yes |
| 1024 | cpu-fused | go | 1.743 | +3.9% | 0.210 | -3.7% | 644 | 5 | yes |

| lang | call µs, median (1 row x 64 centroids, 1 thread) | min - max | overhead vs Rust µs |
|---|---|---|---|
| rust | 215.5 | 197.7 - 225.2 | — |
| python | 202.4 | 199.2 - 213.0 | -13.1 |
| node | 598.7 | 534.2 - 778.2 | +383.3 |
| go | 217.2 | 206.4 - 220.1 | +1.7 |
