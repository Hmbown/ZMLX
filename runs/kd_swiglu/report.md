# Kernel Discovery Report: `swiglu`

- Seed: `0`
- Budget: `20`
- DType: `float16`
- Shape suite: `glm_flash_small`
- Candidates total: `47`
- Benchmarked: `1`

## Top 10 by Latency

| Rank | Candidate | Latency (us) | Speedup vs ref | Max abs err |
|:--|:--|--:|--:|--:|
| 1 | `swiglu_e79b57774f9d99c2` | 195.645 | 1.278 | 0.000e+00 |

## Pareto Frontier (Latency vs Error)

| Candidate | Latency (us) | Max abs err | Max rel err | Speedup |
|:--|--:|--:|--:|--:|
| `swiglu_e79b57774f9d99c2` | 195.645 | 0.000e+00 | 0.000e+00 | 1.278 |

## Knob Notes

- Not enough benchmarked candidates to infer knob effects.
