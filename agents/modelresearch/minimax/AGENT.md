# MiniMax (M2.1) Advisor Agent

Use `agents/modelresearch/TEMPLATE_AGENT.md` as the baseline. This file adds
MiniMax-specific, **repo-verified** details (Exo cards + Exo sharding).

## Bundled model IDs (from Exo model cards)
- `mlx-community/MiniMax-M2.1-3bit`
- `mlx-community/MiniMax-M2.1-8bit`

## Config snapshot (from Exo model cards)

Both bundled MiniMax cards share:
- `n_layers = 61`
- `hidden_size = 3072`
- `supports_tensor = true`

| model_id | n_layers | hidden_size | supports_tensor | storage_size.in_bytes |
|---|---:|---:|---:|---:|
| `mlx-community/MiniMax-M2.1-3bit` | 61 | 3072 | true | 100086644736 |
| `mlx-community/MiniMax-M2.1-8bit` | 61 | 3072 | true | 242986745856 |

## Exo integration notes (verified in this repo)

- Sharding and required module surface are defined in `exo/src/exo/worker/engines/mlx/auto_parallel.py` (`MiniMaxShardingStrategy`):
  - Shards `layer.self_attn.{q_proj,k_proj,v_proj}` and shards-to-all `o_proj`.
  - If `layer.self_attn.use_qk_norm` is set, it additionally shards `q_norm.weight` and `k_norm.weight` by splitting the last dim across ranks.
  - Divides `num_attention_heads` and `num_key_value_heads` by world size.
  - Shards MoE projections in-place under `layer.block_sparse_moe.switch_mlp.{gate_proj,up_proj,down_proj}` and wraps the MoE block with `ShardedQwenMoE` to `all_sum` the output.

## Verified module surface (from Exo sharding code)

Exo’s sharding assumes MiniMax layers expose (names taken directly from `auto_parallel.py`):
- Attention: `layer.self_attn.{q_proj,k_proj,v_proj,o_proj}`, `num_attention_heads`, `num_key_value_heads`, plus optional `use_qk_norm` and `q_norm.weight` / `k_norm.weight`.
- MoE: `layer.block_sparse_moe.switch_mlp.{gate_proj,up_proj,down_proj}`.
