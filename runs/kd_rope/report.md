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
| 1 | `rope_c73769625b5efd6b` | 130.459 | 1.748 | 0.000e+00 |
| 2 | `rope_5e49a0eb05882163` | 133.875 | 1.703 | 0.000e+00 |
| 3 | `rope_981b7e3c51434fe4` | 139.834 | 1.631 | 0.000e+00 |
| 4 | `rope_d8bd5f5b96766e89` | 140.334 | 1.625 | 0.000e+00 |
| 5 | `rope_faa2cc8c9f591b70` | 143.291 | 1.591 | 0.000e+00 |
| 6 | `rope_8e6b355277070755` | 144.105 | 1.582 | 0.000e+00 |
| 7 | `rope_da5cfefdaee0904b` | 157.916 | 1.444 | 0.000e+00 |
| 8 | `rope_81f50de3ebb400a3` | 162.646 | 1.392 | 0.000e+00 |

## Pareto Frontier (Latency vs Error)

| Candidate | Latency (us) | Max abs err | Max rel err | Speedup |
|:--|--:|--:|--:|--:|
| `rope_c73769625b5efd6b` | 130.459 | 0.000e+00 | 0.000e+00 | 1.748 |
| `rope_5e49a0eb05882163` | 133.875 | 0.000e+00 | 0.000e+00 | 1.703 |
| `rope_981b7e3c51434fe4` | 139.834 | 0.000e+00 | 0.000e+00 | 1.631 |
| `rope_d8bd5f5b96766e89` | 140.334 | 0.000e+00 | 0.000e+00 | 1.625 |
| `rope_faa2cc8c9f591b70` | 143.291 | 0.000e+00 | 0.000e+00 | 1.591 |
| `rope_8e6b355277070755` | 144.105 | 0.000e+00 | 0.000e+00 | 1.582 |
| `rope_da5cfefdaee0904b` | 157.916 | 0.000e+00 | 0.000e+00 | 1.444 |
| `rope_81f50de3ebb400a3` | 162.646 | 0.000e+00 | 0.000e+00 | 1.392 |

## Knob Notes

- `launch.threadgroup_x`: top average=384.000, rest average=192.000
