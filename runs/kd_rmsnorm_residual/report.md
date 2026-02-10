# Kernel Discovery Report: `rmsnorm_residual`

- Seed: `0`
- Budget: `20`
- DType: `float16`
- Shape suite: `glm_flash_small`
- Candidates total: `49`
- Benchmarked: `20`

## Top 10 by Latency

| Rank | Candidate | Latency (us) | Speedup vs ref | Max abs err |
|:--|:--|--:|--:|--:|
| 1 | `rmsnorm_residual_3d9e4da0b8a43ea6` | 160.188 | 1.930 | 7.812e-03 |
| 2 | `rmsnorm_residual_b600ea0810a9af0c` | 167.584 | 1.845 | 7.812e-03 |
| 3 | `rmsnorm_residual_1c171a133044a311` | 169.062 | 1.829 | 7.812e-03 |
| 4 | `rmsnorm_residual_efb7abf5f90288b7` | 175.791 | 1.759 | 3.906e-03 |
| 5 | `rmsnorm_residual_302da93503d48525` | 175.791 | 1.759 | 7.812e-03 |
| 6 | `rmsnorm_residual_3fb8d765e6b2a090` | 176.188 | 1.755 | 3.906e-03 |
| 7 | `rmsnorm_residual_9fb2437335e855ee` | 176.416 | 1.753 | 7.812e-03 |
| 8 | `rmsnorm_residual_96eaf3a799e17b29` | 176.895 | 1.756 | 7.812e-03 |
| 9 | `rmsnorm_residual_e4ef568da100c168` | 180.270 | 1.715 | 7.812e-03 |
| 10 | `rmsnorm_residual_5474f82bb7757b69` | 180.771 | 1.710 | 7.812e-03 |

## Pareto Frontier (Latency vs Error)

| Candidate | Latency (us) | Max abs err | Max rel err | Speedup |
|:--|--:|--:|--:|--:|
| `rmsnorm_residual_3d9e4da0b8a43ea6` | 160.188 | 7.812e-03 | 1.276e-03 | 1.930 |
| `rmsnorm_residual_b600ea0810a9af0c` | 167.584 | 7.812e-03 | 3.378e-03 | 1.845 |
| `rmsnorm_residual_1c171a133044a311` | 169.062 | 7.812e-03 | 1.337e-03 | 1.829 |
| `rmsnorm_residual_efb7abf5f90288b7` | 175.791 | 3.906e-03 | 1.855e-03 | 1.759 |
| `rmsnorm_residual_3fb8d765e6b2a090` | 176.188 | 3.906e-03 | 2.415e-03 | 1.755 |

## Knob Notes

- `unroll`: top average=2.400, rest average=1.667
- `use_simd`: top average=0.400, rest average=0.533
- `vec_width`: top average=1.400, rest average=2.067
- `launch.threadgroup_x`: top average=256.000, rest average=320.000
