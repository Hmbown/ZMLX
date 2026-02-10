# Kernel Discovery Report: `swiglu`

- Seed: `0`
- Budget: `20`
- DType: `float16`
- Shape suite: `qwen30b_decode`
- Candidates total: `47`
- Benchmarked: `1`

## Top 10 by Latency

| Rank | Candidate | Latency (us) | Speedup vs ref | Max abs err |
|:--|:--|--:|--:|--:|
| 1 | `swiglu_7470046f4cbdfe5f` | 147.000 | 1.026 | 0.000e+00 |

## Pareto Frontier (Latency vs Error)

| Candidate | Latency (us) | Max abs err | Max rel err | Speedup |
|:--|--:|--:|--:|--:|
| `swiglu_7470046f4cbdfe5f` | 147.000 | 0.000e+00 | 0.000e+00 | 1.026 |

## Knob Notes

- Not enough benchmarked candidates to infer knob effects.
