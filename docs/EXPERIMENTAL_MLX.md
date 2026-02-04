# Custom MLX Primitive (Optional)

This document describes `gather_qmm_swiglu`, a **custom C++ Metal primitive** implemented in `mlx_local/` as an extension to MLX. It is not part of released MLX and must be built locally.

## What this is

`mlx_local/` is a local checkout of upstream MLX (`ml-explore/mlx`, commit `2f324cc`) with ~800 lines of custom C++ and Metal shader code adding the `GatherQMMSwiGLU` primitive. This fuses **gate projection + up projection + SwiGLU activation** for quantized MoE experts into a single GPU dispatch, eliminating multiple kernel launches per expert per layer during decode.

The primitive is exposed as `mx.gather_qmm_swiglu()` in Python when the custom build is active.

## What it does

During MoE decode, each active expert normally requires separate kernel launches for:
1. Dequantize + matmul (gate projection)
2. Dequantize + matmul (up projection)
3. SiLU activation
4. Elementwise multiply (gate * up)

`gather_qmm_swiglu` fuses all four into a single Metal kernel launch per expert. At decode (M=1), where dispatch overhead dominates over compute, this reduces per-layer latency.

## When to use it

- If you want MoE decode speedups on GLM-4.7-Flash or Qwen3-30B-A3B (models where ZMLX auto-skips on stock MLX).
- If you are prototyping fused MLX primitives for potential upstream contribution.

On stock MLX (`pip install mlx`), ZMLX auto-detects that `gather_qmm_swiglu` is unavailable and skips the fused paths. No action needed.

## Set up `mlx_local/`

`mlx_local/` is **not shipped** as part of ZMLX; it is intended as a local-only directory (gitignored) created by cloning MLX and applying a patch.

Recommended:

```bash
bash integrations/mlx_local_integration/setup_mlx_local.sh
```

Manual (equivalent):

```bash
git clone https://github.com/ml-explore/mlx.git mlx_local
cd mlx_local
git checkout 2f324cc3b200700b422db4811ae3ff8bd5bf48b4
git apply <REPO_ROOT>/integrations/mlx_local_integration/gather_qmm_swiglu.patch
```

## Build

```bash
cd mlx_local
python3 setup.py build_ext --inplace
# Limit CPU usage during build if desired:
# CMAKE_BUILD_PARALLEL_LEVEL=4 python3 setup.py build_ext --inplace
```

Then make sure `mlx_local/python` is on your Python path before the stock MLX:

```bash
export PYTHONPATH=<REPO_ROOT>/mlx_local/python:<REPO_ROOT>/src:$PYTHONPATH
python3 -c "import mlx.core as mx; print(hasattr(mx, 'gather_qmm_swiglu'))"  # should print True
```

## Validate

```bash
python3 -m zmlx.validate mlx-community/GLM-4.7-Flash-4bit --max-tokens 128 --runs 5
python3 -m zmlx.validate mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit --max-tokens 128 --runs 5
```

Remove `mlx_local/python` from `PYTHONPATH` to revert to stock MLX.

## Measured results (M4 Max 36 GB)

| Model | Decode (baseline -> patched) | Change | Fidelity |
|:--|--:|--:|:--|
| GLM-4.7-Flash-4bit | 85.8 -> 92.8 tok/s | +8.1% | 128/128 identical |
| Qwen3-30B-A3B-4bit | 117 -> 123 tok/s | +5.5% | 128/128 identical |

## Upstream plan

See [`UPSTREAM_PLAN.md`](../UPSTREAM_PLAN.md). The intent is to contribute `gather_qmm_swiglu` to upstream MLX once it has been validated across more models and hardware.

## Known constraints

- N must be divisible by 8, K by 512.
- Only `transpose=True` and `mode='affine'` are implemented.
- CPU fallback exists but is not optimized (Metal GPU path only).
