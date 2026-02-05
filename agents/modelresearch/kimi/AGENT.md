# Kimi (K2 / K2.5) Advisor Agent

Use `agents/modelresearch/TEMPLATE_AGENT.md` as the baseline. This file adds
Kimi-specific, **repo-verified** details (Exo cards + tokenizer/tool parsing +
sharding expectations).

## Bundled model IDs (from Exo model cards)
- `mlx-community/Kimi-K2-Instruct-4bit`
- `mlx-community/Kimi-K2-Thinking`
- `mlx-community/Kimi-K2.5`
- `inferencerlabs/Kimi-K2.5-MLX-3.6bit`
- `inferencerlabs/Kimi-K2.5-MLX-4.2bit`

## Config snapshot (from Exo model cards)

All bundled Kimi cards share:
- `n_layers = 61`
- `hidden_size = 7168`
- `supports_tensor = true`

| model_id | n_layers | hidden_size | supports_tensor | storage_size.in_bytes |
|---|---:|---:|---:|---:|
| `mlx-community/Kimi-K2-Instruct-4bit` | 61 | 7168 | true | 620622774272 |
| `mlx-community/Kimi-K2-Thinking` | 61 | 7168 | true | 706522120192 |
| `mlx-community/Kimi-K2.5` | 61 | 7168 | true | 662498705408 |
| `inferencerlabs/Kimi-K2.5-MLX-3.6bit` | 61 | 7168 | true | 470011424768 |
| `inferencerlabs/Kimi-K2.5-MLX-4.2bit` | 61 | 7168 | true | 545523496960 |

## Exo integration notes (verified in this repo)

- **Tokenizer + EOS handling**: `exo/src/exo/worker/engines/mlx/utils_mlx.py`
  - `get_eos_token_ids_for_model()` returns `[163585, 163586]` for any model_id containing `kimi-k2`.
  - `load_tokenizer_for_model_id()` has a Kimi-specific path that loads `tokenization_kimi.py` from the model directory and patches imports for transformers 5.x compatibility.
- **Tool calling**: `exo/src/exo/worker/runner/runner.py`
  - `filter_kimi_tokens()` filters out `<|tool_calls_section_begin|>` / `<|tool_calls_section_end|>` from the stream.
  - `patch_kimi_tokenizer()` installs a Kimi-specific tool parser that recognizes the `functions.<name>:<n> <|tool_call_argument_begin|> {...}` format.
- **trust_remote_code**: `exo/src/exo/worker/engines/mlx/constants.py` sets `TRUST_REMOTE_CODE = True` (comment notes Kimi requires it).
- **Sharding / model class availability**: `exo/src/exo/worker/engines/mlx/auto_parallel.py`
  - Exo tries to import `mlx_lm.models.kimi_k25.Model` as `KimiK25Model` (optional; may be missing depending on MLX-LM version).
  - If present, Kimi is treated as DeepSeek-like for tensor sharding via `DeepSeekShardingStrategy`.

## Verified module surface (from Exo sharding code)

If `KimiK25Model` is available and a Kimi model is tensor-sharded, Exo assumes the same DeepSeek-style surface described in:
- `exo/src/exo/worker/engines/mlx/auto_parallel.py` (`DeepSeekShardingStrategy`)
