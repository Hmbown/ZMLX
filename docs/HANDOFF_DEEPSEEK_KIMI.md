# DeepSeek-V3.2 / Kimi-K2.5 MoE router fusion (handoff)

Branch: `deepseek-kimi-moe`

This branch adds a **token-identical** (in current unit tests) fused router for
DeepSeek/Kimi-style MoE gating:

1) `affinity = sigmoid(logits)` (computed in float32)
2) `selection_score = affinity + bias` (**bias affects selection only**)
3) Top‑K selected by `selection_score` with **stable tie‑break** (lower expert id)
4) `weights = affinity[topk] / sum(affinity[topk])`

## What’s implemented

- Fused Metal kernel + Python API:
  - `src/zmlx/kernels/moe.py` → `deepseek_router_topk_sigmoid()`
  - Supports:
    - `Nr ∈ {256, 384}` routed experts
    - `K = 8`
  - Falls back to a pure‑MLX reference implementation when unsupported.

- Opt‑in patch pattern (not part of defaults):
  - `src/zmlx/patch/patterns/deepseek_router.py` (pattern name: `deepseek_router`)
  - Patches DeepSeek gate modules (`MoEGate`) to return `(indices, weights)`
    using the fused router kernel.
  - Conservative guards:
    - `n_group == 1`, `topk_group == 1`
    - `norm_topk_prob == True`
    - `top_k == 8`
    - `n_routed_experts ∈ {256, 384}`

- Unit tests:
  - `tests/test_deepseek_router_topk_sigmoid.py`
  - Verifies exact index selection (stable tie‑break) and close weight match vs
    a pure‑MLX reference.

## How to run tests

```bash
source .venv/bin/activate
pytest -q tests/test_deepseek_router_topk_sigmoid.py
```

## How to validate on a real model

Once you have a loadable MLX‑LM DeepSeek model on suitable hardware:

```bash
source .venv/bin/activate

# Baseline vs patched (greedy token fidelity + throughput)
python -m zmlx.validate <model_id> \
  --patterns deepseek_router moe_mlp swiglu_mlp \
  --runs 3 --max-tokens 200
```

Notes:
- `deepseek_router` only changes gating. `moe_mlp` is still useful to fuse the
  combine step and (when available) fuse SwiGLU for quantized SwitchGLU experts.
- If the model uses grouped routing (`n_group > 1`), the current pattern will
  no‑op; extend the kernel/pattern to match the model’s exact semantics first.

## Next steps (higher ROI)

1) **Grouped routing support** (if DeepSeek/Kimi gate uses `n_group > 1`)
   - Extend `deepseek_router_topk_sigmoid()` to implement the same group masking
     logic used upstream before top‑k.

2) **Combine-side fusion**
   - Add a fused combine kernel for `shared + Σ(w_k * expert_out_k)` with careful
     FMA/dtype semantics (GLM needed a no‑FMA combine to preserve fidelity).

3) **Phase‑2 primitive (biggest win)**
   - Fuse expert down‑projection and stream the weighted accumulation to avoid
     materializing `(T, K, D)` expert outputs at all.

4) **Kimi K2.5 integration**
   - If/when `mlx-lm` adds a `kimi_k25` model class (or a config mapping),
     update `_is_deepseek_gate_module()` in `src/zmlx/patch/patterns/deepseek_router.py`
     to match the new module path.

