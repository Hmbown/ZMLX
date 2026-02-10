# Kernel Discovery Report: `rmsnorm_residual`

- Seed: `0`
- Budget: `20`
- DType: `float16`
- Shape suite: `lfm2_decode`
- Candidates total: `53`
- Benchmarked: `20`

## Top 10 by Latency

| Rank | Candidate | Latency (us) | Speedup vs ref | Max abs err |
|:--|:--|--:|--:|--:|
| 1 | `rmsnorm_residual_5faa7484fddec673` | 123.730 | 1.621 | 3.906e-03 |
| 2 | `rmsnorm_residual_d6959400a3811a21` | 125.562 | 1.597 | 1.953e-03 |
| 3 | `rmsnorm_residual_2e58f6828a2c7547` | 126.250 | 1.589 | 3.906e-03 |
| 4 | `rmsnorm_residual_17b6b039a2360e82` | 127.000 | 1.579 | 1.953e-03 |
| 5 | `rmsnorm_residual_63005cf76643acef` | 127.042 | 1.579 | 1.953e-03 |
| 6 | `rmsnorm_residual_a62e8f2fe01c67f2` | 129.166 | 1.553 | 1.953e-03 |
| 7 | `rmsnorm_residual_b5466501d920cc8b` | 129.459 | 1.549 | 1.953e-03 |
| 8 | `rmsnorm_residual_b448f0bd33a97cf3` | 129.875 | 1.544 | 3.906e-03 |
| 9 | `rmsnorm_residual_88a4cee4a6ab55f7` | 131.041 | 1.531 | 3.906e-03 |
| 10 | `rmsnorm_residual_aff3eca59171ecf1` | 131.375 | 1.527 | 3.906e-03 |

## Pareto Frontier (Latency vs Error)

| Candidate | Latency (us) | Max abs err | Max rel err | Speedup |
|:--|--:|--:|--:|--:|
| `rmsnorm_residual_5faa7484fddec673` | 123.730 | 3.906e-03 | 9.671e-04 | 1.621 |
| `rmsnorm_residual_d6959400a3811a21` | 125.562 | 1.953e-03 | 9.690e-04 | 1.597 |
| `rmsnorm_residual_17b6b039a2360e82` | 127.000 | 1.953e-03 | 9.625e-04 | 1.579 |
| `rmsnorm_residual_63005cf76643acef` | 127.042 | 1.953e-03 | 9.756e-04 | 1.579 |
| `rmsnorm_residual_a62e8f2fe01c67f2` | 129.166 | 1.953e-03 | 9.690e-04 | 1.553 |
| `rmsnorm_residual_b5466501d920cc8b` | 129.459 | 1.953e-03 | 9.747e-04 | 1.549 |
| `rmsnorm_residual_d3f9b2385783f2aa` | 136.188 | 1.953e-03 | 9.737e-04 | 1.473 |
| `rmsnorm_residual_472cdcb08e2db15d` | 139.479 | 1.953e-03 | 9.728e-04 | 1.438 |

## Knob Notes

- `unroll`: top average=1.600, rest average=1.933
- `use_simd`: top average=0.400, rest average=0.467
- `launch.threadgroup_x`: top average=435.200, rest average=230.400
