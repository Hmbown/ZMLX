# GPT-OSS Advisor Agent

Use `agents/modelresearch/TEMPLATE_AGENT.md` as the baseline. This file adds
GPT-OSS-specific, **repo-verified** details (Exo cards + sharding + tool
parsing).

## Evidence pointers
- Model cards: `exo/resources/inference_model_cards/mlx-community--gpt-oss-*.toml`
- Sharding: `exo/src/exo/worker/engines/mlx/auto_parallel.py` (`GptOssShardingStrategy`)
- Tool/thinking parsing: `exo/src/exo/worker/runner/runner.py` (`parse_gpt_oss`)
- ZMLX family detection + excludes: `src/zmlx/patch/__init__.py` (`gpt_oss`)

## Config snapshot (from bundled Exo model cards)
- Bundled model IDs:
  - `mlx-community/gpt-oss-20b-MXFP4-Q8`
  - `mlx-community/gpt-oss-120b-MXFP4-Q8`

| model_id | n_layers | hidden_size | supports_tensor | storage_size.in_bytes |
|---|---:|---:|---:|---:|
| `mlx-community/gpt-oss-20b-MXFP4-Q8` | 24 | 2880 | true | 12025908224 |
| `mlx-community/gpt-oss-120b-MXFP4-Q8` | 36 | 2880 | true | 70652212224 |

## Exo integration notes (verified in this repo)

- Sharding is defined in `exo/src/exo/worker/engines/mlx/auto_parallel.py` (`GptOssShardingStrategy`):
  - Shards `layer.self_attn.{q_proj,k_proj,v_proj}` and shards-to-all `o_proj`.
  - Adjusts head counts and recomputes `num_key_value_groups`.
  - Slices `layer.self_attn.sinks` per-rank.
  - Shards expert projections in-place (`layer.mlp.experts.{gate_proj,up_proj,down_proj}`) and wraps MoE with `ShardedGptOssMoE` to `all_sum` the output.
- Tool + thinking parsing is implemented in `exo/src/exo/worker/runner/runner.py` (`parse_gpt_oss`) using `openai_harmony`.

## Verified module surface (from Exo sharding code)

Exo’s sharding assumes GPT-OSS layers expose (names taken directly from `auto_parallel.py`):
- Attention: `layer.self_attn.{q_proj,k_proj,v_proj,o_proj}`, `num_attention_heads`, `num_key_value_heads`, `sinks`.
- MoE: `layer.mlp.experts.{gate_proj,up_proj,down_proj}`.
