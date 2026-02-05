# DeepSeek (V3 / V3.1 / V3.2) Advisor Agent

Use `agents/modelresearch/TEMPLATE_AGENT.md` as the baseline. This file adds
DeepSeek-specific, **repo-verified** details (Exo model cards + Exo sharding).

## Bundled model IDs (from Exo model cards)
- `mlx-community/DeepSeek-V3-0324-4bit`
- `mlx-community/DeepSeek-V3-0324-5bit`
- `mlx-community/DeepSeek-v3-0324-8bit`
- `mlx-community/DeepSeek-V3-3bit`
- `mlx-community/DeepSeek-V3-3bit-bf16`
- `mlx-community/DeepSeek-V3-4bit`
- `mlx-community/DeepSeek-V3.1-4bit`
- `mlx-community/DeepSeek-V3.1-8bit`
- `mlx-community/DeepSeek-V3.1-Base-4bit`
- `mlx-community/DeepSeek-V3.1-Terminus-4bit`
- `mlx-community/DeepSeek-V3.1-mlx-DQ5_K_M`
- `mlx-community/DeepSeek-V3.2-4bit`
- `mlx-community/DeepSeek-V3.2-8bit`
- `mlx-community/DeepSeek-V3.2-Speciale-4bit`
- `mlx-community/DeepSeek-V3.2-mlx-5bit`
- `mlx-community/DeepSeek-V3.2_bf16`

## Config snapshot (from Exo model cards)

All bundled DeepSeek cards share:
- `n_layers = 61`
- `hidden_size = 7168`
- `supports_tensor = true`

| model_id | n_layers | hidden_size | supports_tensor | storage_size.in_bytes |
|---|---:|---:|---:|---:|
| `mlx-community/DeepSeek-V3-0324-4bit` | 61 | 7168 | true | 377606822912 |
| `mlx-community/DeepSeek-V3-0324-5bit` | 61 | 7168 | true | 461471723520 |
| `mlx-community/DeepSeek-v3-0324-8bit` | 61 | 7168 | true | 754998771712 |
| `mlx-community/DeepSeek-V3-3bit` | 61 | 7168 | true | 335674387456 |
| `mlx-community/DeepSeek-V3-3bit-bf16` | 61 | 7168 | true | 335674387456 |
| `mlx-community/DeepSeek-V3-4bit` | 61 | 7168 | true | 377606822912 |
| `mlx-community/DeepSeek-V3.1-4bit` | 61 | 7168 | true | 405874409472 |
| `mlx-community/DeepSeek-V3.1-8bit` | 61 | 7168 | true | 765577920512 |
| `mlx-community/DeepSeek-V3.1-Base-4bit` | 61 | 7168 | true | 377606852608 |
| `mlx-community/DeepSeek-V3.1-Terminus-4bit` | 61 | 7168 | true | 377606852608 |
| `mlx-community/DeepSeek-V3.1-mlx-DQ5_K_M` | 61 | 7168 | true | 472916152320 |
| `mlx-community/DeepSeek-V3.2-4bit` | 61 | 7168 | true | 378085857792 |
| `mlx-community/DeepSeek-V3.2-8bit` | 61 | 7168 | true | 755956750848 |
| `mlx-community/DeepSeek-V3.2-Speciale-4bit` | 61 | 7168 | true | 378085857792 |
| `mlx-community/DeepSeek-V3.2-mlx-5bit` | 61 | 7168 | true | 462057167360 |
| `mlx-community/DeepSeek-V3.2_bf16` | 61 | 7168 | true | 1343755917824 |

## Exo integration notes (verified in this repo)

- Model cards live in `exo/resources/inference_model_cards/` (see list above).
- Sharding and required module surface are defined in `exo/src/exo/worker/engines/mlx/auto_parallel.py`:
  - Tensor sharding uses `DeepSeekShardingStrategy`.
  - Attention sharding depends on `layer.self_attn.q_lora_rank`:
    - If `None`, shard `layer.self_attn.q_proj`
    - Else, shard `layer.self_attn.q_b_proj`
  - Always shards `layer.self_attn.kv_b_proj`, shards-to-all `layer.self_attn.o_proj`, and sets `layer.self_attn.num_heads //= world_size`.
  - MoE layers shard expert projections in-place and wrap the MoE module with `ShardedDeepseekV3MoE` which `all_sum`s the output across ranks.
- Placement restriction: `exo/src/exo/master/placement.py` blocks **pipeline parallelism** for `mlx-community/DeepSeek-V3.1-8bit`.
- ZMLX experiments: the opt-in DeepSeek/Kimi router fusion is documented in `docs/DEEPSEEK_KIMI_ROUTER_FUSION.md` and `docs/HANDOFF_DEEPSEEK_KIMI.md`.

## Verified module surface (from Exo sharding code)

Exo’s sharding assumes DeepSeek-style layers expose (names taken directly from `auto_parallel.py`):
- Attention:
  - `layer.self_attn.q_lora_rank`
  - `layer.self_attn.q_proj` **or** `layer.self_attn.q_b_proj`
  - `layer.self_attn.kv_b_proj`
  - `layer.self_attn.o_proj`
  - `layer.self_attn.num_heads`
- Dense MLP case:
  - `layer.mlp.gate_proj`, `layer.mlp.up_proj`, `layer.mlp.down_proj`
- MoE case:
  - `layer.mlp.shared_experts.{gate_proj,up_proj,down_proj}`
  - `layer.mlp.switch_mlp.{gate_proj,up_proj,down_proj}`
