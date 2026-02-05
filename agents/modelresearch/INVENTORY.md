# Exo model family inventory (from bundled model cards)

This inventory is derived from:
- `exo/resources/inference_model_cards/*.toml` (text)
- `exo/resources/image_model_cards/*.toml` (image; **deferred** — not needed for current work)

## Text (MLX)
### DeepSeek (V3 / V3.1 / V3.2)
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

### Kimi (K2 / K2.5)
- `mlx-community/Kimi-K2-Instruct-4bit`
- `mlx-community/Kimi-K2-Thinking`
- `mlx-community/Kimi-K2.5`
- `inferencerlabs/Kimi-K2.5-MLX-3.6bit`
- `inferencerlabs/Kimi-K2.5-MLX-4.2bit`

### MiniMax (M2.1)
- `mlx-community/MiniMax-M2.1-3bit`
- `mlx-community/MiniMax-M2.1-8bit`

### Qwen3 (dense + MoE + Next variants)
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

### GLM-4.x (GLM-4.5-Air, GLM-4.7, GLM-4.7-Flash)
- `mlx-community/GLM-4.5-Air-8bit`
- `mlx-community/GLM-4.5-Air-bf16`
- `mlx-community/GLM-4.7-4bit`
- `mlx-community/GLM-4.7-6bit`
- `mlx-community/GLM-4.7-8bit-gs32`
- `mlx-community/GLM-4.7-Flash-4bit`
- `mlx-community/GLM-4.7-Flash-5bit`
- `mlx-community/GLM-4.7-Flash-6bit`
- `mlx-community/GLM-4.7-Flash-8bit`

### Llama 3.x (Meta-Llama 3.1 / Llama 3.2 / Llama 3.3)
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

### GPT-OSS
- `mlx-community/gpt-oss-20b-MXFP4-Q8`
- `mlx-community/gpt-oss-120b-MXFP4-Q8`

### Ministral3 (Mistral-family)
- Exo imports `mlx_lm.models.ministral3.Model` and treats it as Llama-like for sharding, but there are currently no bundled `exo/resources/inference_model_cards/*.toml` for Ministral3.

Notes:
- This list is “what Exo ships model cards for”. A family can still be loadable via MLX-LM without a bundled card, but it won’t be discoverable from the card inventory.
