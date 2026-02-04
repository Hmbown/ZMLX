# exo Integration (GLM/Qwen MoE decode wins)

This repo includes an `exo/` checkout with an opt-in hook that applies ZMLX fused-kernel patches to exo’s MLX engine.

## Quickstart (single node, best GLM decode win)

The best GLM-4.7-Flash decode win depends on the fused primitive `mx.gather_qmm_swiglu`, which is **not** exposed in released MLX 0.30.4/0.30.5 as of 2026-02-04. The repo’s `mlx_local/` dev build exposes it.

Start exo with:

```bash
cd <REPO_ROOT>/exo
export PYTHONPATH=<REPO_ROOT>/mlx_local/python:<REPO_ROOT>/src:$PYTHONPATH
export EXO_ZMLX=1
uv run -p 3.14 exo --fast-synch
```

Notes:
- `-p 3.14` is required if your `mlx_local/python/mlx/core.*.so` is built for CPython 3.14 (e.g. `cpython-314`). Match this to your local build.
- `--fast-synch` is an exo optimization that forces `MLX_METAL_FAST_SYNCH=1` in runner processes.

Place a model instance (downloads + loads the model):

```bash
curl http://localhost:52415/place_instance \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"mlx-community/GLM-4.7-Flash-4bit"}'
```

Benchmark (returns `generation_stats.generation_tps`):

```bash
curl http://localhost:52415/bench/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/GLM-4.7-Flash-4bit","messages":[{"role":"user","content":"Hello"}],"max_tokens":128,"temperature":0,"top_p":1}'
```

## Baseline vs patched (what to expect)

- Baseline exo: run without `EXO_ZMLX=1`.
- Patched exo: run with `EXO_ZMLX=1`.

On the reference machine (M4 Max), `EXO_ZMLX=1` improved GLM generation throughput from ~84 tok/s → ~92 tok/s in exo bench mode (token parity preserved by the underlying ZMLX patches; verify with `python -m zmlx.validate`).

## Stock MLX behavior

If `mx.gather_qmm_swiglu` is unavailable (released MLX builds), ZMLX will skip GLM/Qwen MoE fused decode paths by default to avoid regressions. You can still:
- Use `smart_patch`/`zmlx.validate` to benchmark what helps in your environment.
- Keep ZMLX enabled for other model families/patterns that don’t require this primitive.

