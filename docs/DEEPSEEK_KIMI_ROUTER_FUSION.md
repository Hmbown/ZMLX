# DeepSeek-style MoE router fusion (DeepSeek-V3.2 / Kimi-K2.5) — scaffolding

This work area adds an **opt-in** router kernel + patch pattern for MLX-LM
DeepSeek-style MoE routing (used by DeepSeek-V3/V3.2 and Kimi-K2.5).

## What is fused

Given router logits `logits` with shape `(..., Nr)` and expert bias `bias` with
shape `(Nr,)`, routing is:

1. `affinity = sigmoid(logits)` *(float32)*
2. `selection_score = affinity + bias` *(bias only affects which experts are chosen)*
3. `topk = top_k(selection_score)` with stable tie-break (lower expert index wins)
4. `weights = affinity[topk] / sum(affinity[topk])`

Current fused kernel support:
- `Nr` (routed experts): **256** or **384**
- `K` (top-k): **8**
- No group selection (`n_group == 1`)
- Normalized top-k weights (`norm_topk_prob == True`)

Kernel entrypoint: `zmlx.kernels.moe.deepseek_router_topk_sigmoid`.

Patch pattern (opt-in): `deepseek_router` (patches MLX-LM `MoEGate` for
`deepseek_v3` / `deepseek_v32` only).

## Quick correctness check

Run the unit test (requires Metal backend):

```bash
pytest -q tests/test_deepseek_router_topk_sigmoid.py
```

## Validate on a real model

Use `zmlx.validate` to check greedy token fidelity + throughput on your machine:

```bash
python -m zmlx.validate <HF_MODEL_ID> \
  --patterns deepseek_router moe_mlp \
  --runs 3 --max-tokens 200
```

Notes:
- `deepseek_router` is **not** part of default presets; you must pass it in
  `--patterns` (or `patch(..., patterns=[...])`) to enable.
- If the model uses group routing (`n_group > 1`) or a different `top_k`, this
  pattern currently no-ops (falls back to the model’s original routing).

## Next steps

- Extend the fused router kernel to support DeepSeek group selection
  (`n_group > 1`, `topk_group < n_group`) while preserving exact MLX-LM semantics.
- Add a benchmark runner that writes a repro capsule under
  `benchmarks/repro_capsules/` for DeepSeek/Kimi models (so any perf claims are
  fully reproducible).
