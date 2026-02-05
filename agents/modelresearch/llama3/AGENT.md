# Llama 3.x Advisor Agent

Use `agents/modelresearch/TEMPLATE_AGENT.md` as the baseline. This file adds
Llama-specific, **repo-verified** details (Exo model cards + Exo sharding).

## Bundled model IDs (from Exo model cards)
- `mlx-community/Llama-3.2-1B-Instruct-4bit`
- `mlx-community/Llama-3.2-3B-Instruct-4bit`
- `mlx-community/Llama-3.2-3B-Instruct-8bit`
- `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`
- `mlx-community/Meta-Llama-3.1-8B-Instruct-8bit`
- `mlx-community/Meta-Llama-3.1-8B-Instruct-bf16`
- `mlx-community/Meta-Llama-3.1-70B-Instruct-4bit`
- `mlx-community/Llama-3.3-70B-Instruct-4bit`
- `mlx-community/Llama-3.3-70B-Instruct-8bit`
- `mlx-community/llama-3.3-70b-instruct-fp16`

## Config snapshot (from Exo model cards)

| model_id | n_layers | hidden_size | supports_tensor | storage_size.in_bytes |
|---|---:|---:|---:|---:|
| `mlx-community/Llama-3.2-1B-Instruct-4bit` | 16 | 2048 | true | 729808896 |
| `mlx-community/Llama-3.2-3B-Instruct-4bit` | 28 | 3072 | true | 1863319552 |
| `mlx-community/Llama-3.2-3B-Instruct-8bit` | 28 | 3072 | true | 3501195264 |
| `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` | 32 | 4096 | true | 4637851648 |
| `mlx-community/Meta-Llama-3.1-8B-Instruct-8bit` | 32 | 4096 | true | 8954839040 |
| `mlx-community/Meta-Llama-3.1-8B-Instruct-bf16` | 32 | 4096 | true | 16882073600 |
| `mlx-community/Meta-Llama-3.1-70B-Instruct-4bit` | 80 | 8192 | true | 40652242944 |
| `mlx-community/Llama-3.3-70B-Instruct-4bit` | 80 | 8192 | true | 40652242944 |
| `mlx-community/Llama-3.3-70B-Instruct-8bit` | 80 | 8192 | true | 76799803392 |
| `mlx-community/llama-3.3-70b-instruct-fp16` | 80 | 8192 | true | 144383672320 |

## Exo integration notes (verified in this repo)

- Sharding and required module surface are defined in `exo/src/exo/worker/engines/mlx/auto_parallel.py`:
  - Tensor sharding uses `LlamaShardingStrategy` for both `mlx_lm.models.llama.Model` and `mlx_lm.models.ministral3.Model`.
  - Shards `layer.self_attn.{q_proj,k_proj,v_proj}` and shards-to-all `o_proj`.
  - Divides `n_heads` and `n_kv_heads` by world size.
  - Shards MLP projections (`gate_proj`, `up_proj`, `down_proj`) in the standard Llama pattern.

## Verified module surface (from Exo sharding code)

Exo’s sharding assumes Llama-like layers expose (names taken directly from `auto_parallel.py`):
- Attention: `layer.self_attn.{q_proj,k_proj,v_proj,o_proj}`, `n_heads`, and optionally `n_kv_heads`.
- MLP: `layer.mlp.{gate_proj,up_proj,down_proj}`.
