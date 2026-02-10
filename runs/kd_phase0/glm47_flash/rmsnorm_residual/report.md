# Kernel Discovery Report: `rmsnorm_residual`

- Seed: `0`
- Budget: `20`
- DType: `float16`
- Shape suite: `glm_flash_small`
- Candidates total: `47`
- Benchmarked: `20`

## Top 10 by Latency

| Rank | Candidate | Latency (us) | Speedup vs ref | Max abs err |
|:--|:--|--:|--:|--:|
| 1 | `rmsnorm_residual_9c80e02fae5cc00b` | 141.938 | 2.457 | 7.812e-03 |
| 2 | `rmsnorm_residual_9ee604c83d384850` | 142.062 | 2.455 | 3.906e-03 |
| 3 | `rmsnorm_residual_9fb2437335e855ee` | 152.041 | 2.294 | 7.812e-03 |
| 4 | `rmsnorm_residual_66abd3cdeb4d8d71` | 152.083 | 2.293 | 7.812e-03 |
| 5 | `rmsnorm_residual_4144d2f97a190b25` | 152.875 | 2.281 | 7.812e-03 |
| 6 | `rmsnorm_residual_f385a0b77fd7ac03` | 153.188 | 2.277 | 7.812e-03 |
| 7 | `rmsnorm_residual_8fe75929551945a4` | 154.438 | 2.258 | 7.812e-03 |
| 8 | `rmsnorm_residual_5474f82bb7757b69` | 156.750 | 2.225 | 7.812e-03 |
| 9 | `rmsnorm_residual_439aa7830bf0627e` | 157.937 | 2.208 | 7.812e-03 |
| 10 | `rmsnorm_residual_b373cc82e0571f30` | 159.687 | 2.184 | 7.812e-03 |

## Pareto Frontier (Latency vs Error)

| Candidate | Latency (us) | Max abs err | Max rel err | Speedup |
|:--|--:|--:|--:|--:|
| `rmsnorm_residual_9c80e02fae5cc00b` | 141.938 | 7.812e-03 | 1.580e-03 | 2.457 |
| `rmsnorm_residual_9ee604c83d384850` | 142.062 | 3.906e-03 | 5.814e-03 | 2.455 |
| `rmsnorm_residual_efb7abf5f90288b7` | 163.729 | 3.906e-03 | 1.855e-03 | 2.130 |

## Knob Notes

- `unroll`: top average=1.600, rest average=1.667
- `use_simd`: top average=0.600, rest average=0.467
- `vec_width`: top average=2.600, rest average=1.867
- `launch.threadgroup_x`: top average=256.000, rest average=311.467
