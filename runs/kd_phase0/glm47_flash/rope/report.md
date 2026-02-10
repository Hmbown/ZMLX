# Kernel Discovery Report: `rope`

- Seed: `0`
- Budget: `8`
- DType: `float16`
- Shape suite: `glm_flash_small`
- Candidates total: `8`
- Benchmarked: `8`

## Top 10 by Latency

| Rank | Candidate | Latency (us) | Speedup vs ref | Max abs err |
|:--|:--|--:|--:|--:|
| 1 | `rope_981b7e3c51434fe4` | 129.063 | 1.639 | 0.000e+00 |
| 2 | `rope_8e6b355277070755` | 130.458 | 1.621 | 0.000e+00 |
| 3 | `rope_5e49a0eb05882163` | 130.584 | 1.620 | 0.000e+00 |
| 4 | `rope_c73769625b5efd6b` | 131.333 | 1.611 | 0.000e+00 |
| 5 | `rope_d8bd5f5b96766e89` | 132.375 | 1.598 | 0.000e+00 |
| 6 | `rope_faa2cc8c9f591b70` | 140.291 | 1.508 | 0.000e+00 |
| 7 | `rope_da5cfefdaee0904b` | 144.542 | 1.463 | 0.000e+00 |
| 8 | `rope_81f50de3ebb400a3` | 152.062 | 1.392 | 0.000e+00 |

## Pareto Frontier (Latency vs Error)

| Candidate | Latency (us) | Max abs err | Max rel err | Speedup |
|:--|--:|--:|--:|--:|
| `rope_981b7e3c51434fe4` | 129.063 | 0.000e+00 | 0.000e+00 | 1.639 |
| `rope_8e6b355277070755` | 130.458 | 0.000e+00 | 0.000e+00 | 1.621 |
| `rope_5e49a0eb05882163` | 130.584 | 0.000e+00 | 0.000e+00 | 1.620 |
| `rope_c73769625b5efd6b` | 131.333 | 0.000e+00 | 0.000e+00 | 1.611 |
| `rope_d8bd5f5b96766e89` | 132.375 | 0.000e+00 | 0.000e+00 | 1.598 |
| `rope_faa2cc8c9f591b70` | 140.291 | 0.000e+00 | 0.000e+00 | 1.508 |
| `rope_da5cfefdaee0904b` | 144.542 | 0.000e+00 | 0.000e+00 | 1.463 |
| `rope_81f50de3ebb400a3` | 152.062 | 0.000e+00 | 0.000e+00 | 1.392 |

## Knob Notes

- `launch.threadgroup_x`: top average=64.000, rest average=298.667
