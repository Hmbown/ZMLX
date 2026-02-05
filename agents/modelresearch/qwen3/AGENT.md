# Qwen3 Advisor Agent (dense + MoE + Next variants)

Use `agents/modelresearch/TEMPLATE_AGENT.md` as the baseline. This file adds
Qwen3-specific, **repo-verified** details (Exo model cards + Exo sharding +
ZMLX family excludes).

## Bundled model IDs (from Exo model cards)
- `mlx-community/Qwen3-0.6B-4bit`
- `mlx-community/Qwen3-0.6B-8bit`
- `mlx-community/Qwen3-30B-A3B-4bit`
- `mlx-community/Qwen3-30B-A3B-8bit`
- `mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit`
- `mlx-community/Qwen3-235B-A22B-Instruct-2507-8bit`
- `mlx-community/Qwen3-Coder-480B-A35B-Instruct-4bit`
- `mlx-community/Qwen3-Coder-480B-A35B-Instruct-8bit`
- `mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit`
- `mlx-community/Qwen3-Next-80B-A3B-Instruct-8bit`
- `mlx-community/Qwen3-Next-80B-A3B-Thinking-4bit`
- `mlx-community/Qwen3-Next-80B-A3B-Thinking-8bit`

## Config snapshot (from Exo model cards)

| model_id | n_layers | hidden_size | supports_tensor | storage_size.in_bytes |
|---|---:|---:|---:|---:|
| `mlx-community/Qwen3-0.6B-4bit` | 28 | 1024 | false | 342884352 |
| `mlx-community/Qwen3-0.6B-8bit` | 28 | 1024 | false | 698351616 |
| `mlx-community/Qwen3-30B-A3B-4bit` | 48 | 2048 | true | 17612931072 |
| `mlx-community/Qwen3-30B-A3B-8bit` | 48 | 2048 | true | 33279705088 |
| `mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit` | 94 | 4096 | true | 141733920768 |
| `mlx-community/Qwen3-235B-A22B-Instruct-2507-8bit` | 94 | 4096 | true | 268435456000 |
| `mlx-community/Qwen3-Coder-480B-A35B-Instruct-4bit` | 62 | 6144 | true | 289910292480 |
| `mlx-community/Qwen3-Coder-480B-A35B-Instruct-8bit` | 62 | 6144 | true | 579820584960 |
| `mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit` | 48 | 2048 | true | 46976204800 |
| `mlx-community/Qwen3-Next-80B-A3B-Instruct-8bit` | 48 | 2048 | true | 88814387200 |
| `mlx-community/Qwen3-Next-80B-A3B-Thinking-4bit` | 48 | 2048 | true | 47080074240 |
| `mlx-community/Qwen3-Next-80B-A3B-Thinking-8bit` | 48 | 2048 | true | 88814387200 |

## Exo integration notes (verified in this repo)

- Model cards live in `exo/resources/inference_model_cards/` (see list above).
- Sharding and required module surface are defined in `exo/src/exo/worker/engines/mlx/auto_parallel.py`:
  - Tensor sharding uses `QwenShardingStrategy` for:
    - `mlx_lm.models.qwen3_moe.Model` (Qwen3 MoE)
    - `mlx_lm.models.qwen3_next.Model` (Qwen3-Next)
    - `mlx_lm.models.glm4_moe.Model` (GLM-4 MoE; shares the same switch-moe surface)
  - MoE blocks are detected by `isinstance(layer.mlp, (Qwen3MoeSparseMoeBlock, MoE, Qwen3NextSparseMoeBlock))` and then sharded **in-place** (gate/up/down) and wrapped with `ShardedQwenMoE` to `all_sum` the output.
  - When pipeline-parallel layers are “shrunk”, `_set_layers()` updates `num_hidden_layers` for `Qwen3MoeModel`.
- ZMLX family detection + safety excludes live in `src/zmlx/patch/__init__.py` (see `_model_family`, `_FIDELITY_EXCLUDES`, `_PERF_EXCLUDES`).

## Verified module surface (from Exo sharding code)

Exo’s sharding assumes Qwen-like layers expose (names taken directly from `auto_parallel.py`):
- Attention: `layer.self_attn.{q_proj,k_proj,v_proj,o_proj}`, plus `layer.self_attn.{n_heads,n_kv_heads}`.
- Dense MLP case: `layer.mlp.{gate_proj,up_proj,down_proj}`.
- MoE case: `layer.mlp.switch_mlp.{gate_proj,up_proj,down_proj}`.
