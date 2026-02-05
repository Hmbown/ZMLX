# GLM-4.x Advisor Agent (GLM-4.5-Air / GLM-4.7 / GLM-4.7-Flash)

Use `agents/modelresearch/TEMPLATE_AGENT.md` as the baseline. This file adds
GLM-specific, **repo-verified** details (Exo cards + tokenizer/thinking/tool
handling + sharding expectations + ZMLX hooks).

## Bundled model IDs (from Exo model cards)
- `mlx-community/GLM-4.5-Air-8bit`
- `mlx-community/GLM-4.5-Air-bf16`
- `mlx-community/GLM-4.7-4bit`
- `mlx-community/GLM-4.7-6bit`
- `mlx-community/GLM-4.7-8bit-gs32`
- `mlx-community/GLM-4.7-Flash-4bit`
- `mlx-community/GLM-4.7-Flash-5bit`
- `mlx-community/GLM-4.7-Flash-6bit`
- `mlx-community/GLM-4.7-Flash-8bit`

## Config snapshot (from Exo model cards)

| model_id | n_layers | hidden_size | supports_tensor | storage_size.in_bytes |
|---|---:|---:|---:|---:|
| `mlx-community/GLM-4.5-Air-8bit` | 46 | 4096 | false | 122406567936 |
| `mlx-community/GLM-4.5-Air-bf16` | 46 | 4096 | true | 229780750336 |
| `mlx-community/GLM-4.7-4bit` | 91 | 5120 | true | 198556925568 |
| `mlx-community/GLM-4.7-6bit` | 91 | 5120 | true | 286737579648 |
| `mlx-community/GLM-4.7-8bit-gs32` | 91 | 5120 | true | 396963397248 |
| `mlx-community/GLM-4.7-Flash-4bit` | 47 | 2048 | true | 19327352832 |
| `mlx-community/GLM-4.7-Flash-5bit` | 47 | 2048 | true | 22548578304 |
| `mlx-community/GLM-4.7-Flash-6bit` | 47 | 2048 | true | 26843545600 |
| `mlx-community/GLM-4.7-Flash-8bit` | 47 | 2048 | true | 34359738368 |

## Exo integration notes (verified in this repo)

- **EOS overrides**: `exo/src/exo/worker/engines/mlx/utils_mlx.py`
  - `get_eos_token_ids_for_model()` returns:
    - for `glm-4.7-flash`: `[154820, 154827, 154829]`
    - for other `glm`: `[151336, 151329, 151338]`
- **Thinking + tool parsing**: `exo/src/exo/worker/runner/runner.py`
  - `parse_thinking_models()` prepends the model’s `think_start` token/text to the stream for “thinking-tag” models (GLM-4.7 is noted in the docstring).
  - `patch_glm_tokenizer()` installs a more robust GLM tool parser (guards regex failures).
- **Sharding**: `exo/src/exo/worker/engines/mlx/auto_parallel.py`
  - `GLM4MoeLiteModel` uses `GLM4MoeLiteShardingStrategy` (LoRA-aware Q projection sharding; `embed_q`/`unembed_out` head slicing; MoE sharding for `shared_experts` + `switch_mlp`).
  - `Glm4MoeModel` is handled by `QwenShardingStrategy` (MoE sharding via `switch_mlp` and an `all_sum` wrapper).
- **ZMLX patching**: `src/zmlx/patch/__init__.py` contains GLM-aware safety excludes and model-family detection.

## Verified module surface (from Exo sharding code)

Exo’s sharding logic assumes GLM layers expose (names taken directly from `auto_parallel.py`):
- Attention:
  - `layer.self_attn.q_lora_rank`
  - `layer.self_attn.q_proj` **or** `layer.self_attn.q_b_proj`
  - `layer.self_attn.o_proj`
  - `layer.self_attn.num_heads`
  - For `GLM4MoeLiteModel` specifically: `layer.self_attn.embed_q.apply(...)` and `layer.self_attn.unembed_out.apply(...)`
- Dense MLP case:
  - `layer.mlp.gate_proj`, `layer.mlp.up_proj`, `layer.mlp.down_proj`
- MoE case:
  - `layer.mlp.switch_mlp.{gate_proj,up_proj,down_proj}`
  - Optional: `layer.mlp.shared_experts.{gate_proj,up_proj,down_proj}`
