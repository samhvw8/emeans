| k | backend | lang | train s (median) | vs Rust | assign s (median) | vs Rust | peak RSS MB | runs | = Rust |
|---|---|---|---|---|---|---|---|---|---|
| 64 | cpu-accelerate | rust | 0.331 | — | 0.028 | — | 634 | 3 | yes |
| 64 | cpu-accelerate | python | 0.345 | +4.1% | 0.023 | -17.7% | 659 | 3 | yes |
| 64 | cpu-accelerate | node | 0.341 | +3.0% | 0.024 | -11.6% | 690 | 3 | yes |
| 64 | cpu-accelerate | go | 0.327 | -1.3% | 0.022 | -18.9% | 643 | 3 | yes |
| 64 | metal-fused | rust | 0.355 | — | 0.135 | — | 1274 | 3 | yes |
| 64 | metal-fused | python | 0.308 | -13.5% | 0.127 | -5.9% | 1300 | 3 | yes |
| 64 | metal-fused | node | 0.344 | -3.2% | 0.171 | +26.6% | 1293 | 3 | yes |
| 64 | metal-fused | go | 0.335 | -5.7% | 0.135 | +0.2% | 1271 | 3 | yes |
| 256 | cpu-accelerate | rust | 0.432 | — | 0.047 | — | 671 | 3 | yes |
| 256 | cpu-accelerate | python | 0.445 | +3.1% | 0.045 | -4.5% | 699 | 3 | yes |
| 256 | cpu-accelerate | node | 0.432 | -0.1% | 0.046 | -1.5% | 728 | 3 | yes |
| 256 | cpu-accelerate | go | 0.435 | +0.8% | 0.047 | +0.4% | 681 | 3 | yes |
| 256 | metal-fused | rust | 0.468 | — | 0.178 | — | 1264 | 3 | yes |
| 256 | metal-fused | python | 0.401 | -14.3% | 0.148 | -16.7% | 1320 | 3 | yes |
| 256 | metal-fused | node | 0.387 | -17.3% | 0.142 | -20.0% | 1343 | 3 | yes |
| 256 | metal-fused | go | 0.377 | -19.4% | 0.141 | -20.4% | 1301 | 3 | yes |
| 1024 | cpu-accelerate | rust | 1.065 | — | 0.148 | — | 742 | 3 | yes |
| 1024 | cpu-accelerate | python | 1.036 | -2.7% | 0.153 | +3.1% | 767 | 3 | yes |
| 1024 | cpu-accelerate | node | 0.993 | -6.8% | 0.146 | -1.4% | 798 | 3 | yes |
| 1024 | cpu-accelerate | go | 0.997 | -6.3% | 0.156 | +5.0% | 752 | 3 | yes |
| 1024 | metal-fused | rust | 0.844 | — | 0.242 | — | 1318 | 3 | yes |
| 1024 | metal-fused | python | 0.828 | -1.8% | 0.264 | +9.4% | 1332 | 3 | yes |
| 1024 | metal-fused | node | 0.820 | -2.7% | 0.254 | +5.1% | 1352 | 3 | yes |
| 1024 | metal-fused | go | 0.829 | -1.8% | 0.239 | -1.2% | 1320 | 3 | yes |

| lang | call µs (1 row x 64 centroids, 1 thread) | overhead vs Rust µs |
|---|---|---|
| rust | 4.1 | — |
| python | 3.2 | -0.9 |
| node | 100.3 | +96.2 |
| go | 5.3 | +1.2 |
