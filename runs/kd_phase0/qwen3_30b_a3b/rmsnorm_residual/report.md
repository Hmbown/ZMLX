# Kernel Discovery Report: `rmsnorm_residual`

- Seed: `0`
- Budget: `20`
- DType: `float16`
- Shape suite: `qwen30b_decode`
- Candidates total: `49`
- Benchmarked: `20`

## Top 10 by Latency

| Rank | Candidate | Latency (us) | Speedup vs ref | Max abs err |
|:--|:--|--:|--:|--:|
| 1 | `rmsnorm_residual_544401a2e05792b8` | 124.250 | 1.662 | 3.906e-03 |
| 2 | `rmsnorm_residual_a62e8f2fe01c67f2` | 125.126 | 1.650 | 1.953e-03 |
| 3 | `rmsnorm_residual_a0667b918a02f1ad` | 125.459 | 1.646 | 3.906e-03 |
| 4 | `rmsnorm_residual_7374411585263db3` | 125.958 | 1.639 | 3.906e-03 |
| 5 | `rmsnorm_residual_6c0648e4c6a2aec5` | 126.666 | 1.630 | 1.953e-03 |
| 6 | `rmsnorm_residual_472cdcb08e2db15d` | 127.062 | 1.625 | 1.953e-03 |
| 7 | `rmsnorm_residual_d6959400a3811a21` | 127.438 | 1.620 | 1.953e-03 |
| 8 | `rmsnorm_residual_006d2b315ce8ae94` | 129.625 | 1.593 | 3.906e-03 |
| 9 | `rmsnorm_residual_b5466501d920cc8b` | 132.916 | 1.554 | 1.953e-03 |
| 10 | `rmsnorm_residual_2a06462f2651e036` | 133.853 | 1.543 | 3.906e-03 |

## Pareto Frontier (Latency vs Error)

| Candidate | Latency (us) | Max abs err | Max rel err | Speedup |
|:--|--:|--:|--:|--:|
| `rmsnorm_residual_544401a2e05792b8` | 124.250 | 3.906e-03 | 9.699e-04 | 1.662 |
| `rmsnorm_residual_a62e8f2fe01c67f2` | 125.126 | 1.953e-03 | 9.690e-04 | 1.650 |
| `rmsnorm_residual_6c0648e4c6a2aec5` | 126.666 | 1.953e-03 | 9.756e-04 | 1.630 |
| `rmsnorm_residual_472cdcb08e2db15d` | 127.062 | 1.953e-03 | 9.728e-04 | 1.625 |
| `rmsnorm_residual_d6959400a3811a21` | 127.438 | 1.953e-03 | 9.690e-04 | 1.620 |
| `rmsnorm_residual_b5466501d920cc8b` | 132.916 | 1.953e-03 | 9.718e-04 | 1.554 |
| `rmsnorm_residual_63005cf76643acef` | 135.959 | 1.953e-03 | 9.756e-04 | 1.519 |
| `rmsnorm_residual_d3f9b2385783f2aa` | 142.771 | 1.953e-03 | 9.551e-04 | 1.446 |

## Knob Notes

- `unroll`: top average=1.600, rest average=1.733
- `use_simd`: top average=0.600, rest average=0.533
- `vec_width`: top average=1.600, rest average=2.133
- `launch.threadgroup_x`: top average=358.400, rest average=247.467
