# ZMLX Ollama-Compatible Server

Drop-in OpenAI-compatible API server powered by `mlx-lm` with ZMLX kernel patches automatically applied.

## Why This Exists

Ollama 0.19+ added MLX support, but its MLX backend is written in Go with CGo
bindings to MLX-C. There is no Python layer, so ZMLX's `nn.Module`-level
patching cannot hook into Ollama's inference pipeline directly.

This server provides the same OpenAI-compatible API that tools like Open WebUI,
Continue, LangChain, and others expect — but runs through `mlx-lm` with ZMLX
patches for fused Metal kernel speedups.

## Quick Start

```bash
# From the ZMLX repo root (venv activated)
python integrations/ollama_compat/serve.py \
    --model /path/to/Qwen3.5-35B-A3B-MLX-4bit \
    --port 8080
```

The server loads the model, applies `zmlx.patch()` (auto-detecting the right
patterns), and starts serving on the specified port.

## Usage

```bash
# Chat completion
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default_model",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 200,
        "temperature": 0.7
    }'

# Streaming
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default_model",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 200,
        "stream": true
    }'
```

**Note:** Use `"model": "default_model"` to refer to the preloaded model. You
can also pass any HuggingFace model ID or local path, and the server will load
and patch it on-the-fly.

## Connecting Tools

### Open WebUI
Set the OpenAI API base URL to `http://localhost:8080/v1`.

### Continue (VS Code)
```json
{
    "models": [{
        "title": "Qwen3.5-35B (ZMLX)",
        "provider": "openai",
        "model": "default_model",
        "apiBase": "http://localhost:8080/v1"
    }]
}
```

### LangChain
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(base_url="http://localhost:8080/v1", model="default_model")
```

## Verified Models

| Model | Patches Applied | Speedup |
|:------|:----------------|:--------|
| Qwen3.5-35B-A3B-MLX-4bit | 70 (30 deltanet + 40 moe_mlp) | ~1-2% decode |
| LFM2-8B-A1B-MLX-4bit | moe_mlp + swiglu_mlp | ~9-12% decode |
| LFM2-24B-A2B-MLX-4bit | moe_mlp (D-SIMD gate) | ~7% decode |

## CLI Options

All `mlx-lm` server options are supported:

```
--model MODEL          HuggingFace ID or local path (required)
--port PORT            Server port (default: 8080)
--host HOST            Bind address (default: 127.0.0.1)
--adapter-path PATH    LoRA adapter path
--max-tokens N         Default max tokens (default: 4096)
--temp FLOAT           Default temperature (default: 0.0)
--trust-remote-code    Trust remote tokenizer code
```
