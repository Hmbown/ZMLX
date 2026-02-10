# Kernel Discovery (Hamiltonian-guided)

`zmlx.kd` adds an additive kernel-discovery workflow for MLX Metal kernels.

## Goals

- Generate candidate kernels + launch configs for target ops.
- Validate correctness against MLX references.
- Benchmark candidates with explicit eval/synchronization boundaries.
- Use graph-guided scheduling (kNN + union-of-cycles tours) for diversity.
- Emit reproducible artifacts (`run.ndjson`, `report.md`, `best_kernels.json`).
- Optionally load pinned winners at runtime via `ZMLX_USE_DISCOVERED_KERNELS=1`.

## Phase 0 Targets

Discovery starts with fused boundaries where wins are likelier than standalone MLX fast ops:

- `rmsnorm_residual` (add + RMSNorm)
- `rope` (decode RoPE + Q/K pack helper)
- `swiglu` (fused gate/up activation path)

Standalone `rmsnorm` is also available for baseline comparison.

## Model Coverage Setup (No Inference)

Phase-0 setup coverage in this repo is evidence-driven from:

- `benchmarks/matrix.jsonl`
- `benchmarks/repro_capsules/`

Tracked model coverage + commands live in:

- `configs/kd_phase0_model_manifest.json`

Current per-family suites:

- `glm`: `rmsnorm_residual`, `rope`, `swiglu` using `glm_flash_small`
- `qwen`: `rmsnorm_residual`, `swiglu` using `qwen30b_decode`
- `lfm`: `rmsnorm_residual`, `swiglu` using `lfm2_decode`

Note: `rope` in `zmlx.kd` currently targets the GLM MLA decode boundary (`q_nope`/`q_rope` + KV concat), so it is not selected for Qwen/LFM setup runs.

## CLI

Use either script form or module form:

```bash
python -m zmlx.kd run --op rmsnorm_residual --budget 200 --seed 0 --out runs/kd_rmsnorm_residual
python -m zmlx.kd run --op swiglu --budget 200 --shapes glm_flash_small --out runs/kd_swiglu
python -m zmlx.kd run --op rope --budget 200 --out runs/kd_rope
zmlx-kernel-discover run --op rmsnorm --budget 50 --seed 0 --out runs/kd_rmsnorm
python -m zmlx.kd report --run runs/kd_rmsnorm
python -m zmlx.kd install --run runs/kd_rmsnorm --output configs/discovered_kernels.json
python -m zmlx.kd pack --runs runs/kd_rmsnorm_residual runs/kd_swiglu runs/kd_rope --out kernelpacks/applegpu_g16s.json
python -m zmlx.kd suggest-shapes --model-id mlx-community/GLM-4.7-Flash-4bit
python -m zmlx.kd run --op rope --model-id mlx-community/GLM-4.7-Flash-4bit --budget 200 --out runs/kd_rope_glm47
```

`pack` is intended for multi-node/cluster aggregation: each node can publish a run,
and `pack` merges all `best_kernels.json` payloads by runtime key while keeping the
lowest latency candidate for duplicate keys.

### Model-Aware Shapes from Hugging Face

To avoid hand-maintained shape suites, `zmlx.kd` can derive shapes from a model's
Hugging Face `config.json`.

- `run --model-id ...` automatically derives shape signatures for the selected op
  when `--shapes` is omitted (or set to `auto`).
- `suggest-shapes --model-id ...` prints or writes derived suites without running discovery.
- `--decode-rows 1,2,4` controls the decode row variants used in derived suites.
- `--local-files-only` restricts config resolution to local Hugging Face cache.

## Artifacts

Each run directory contains:

- `run_meta.json`: run header (seed, op, runtime fingerprint)
- `run.ndjson`: one record per evaluation step
- `report.md`: top candidates + Pareto frontier
- `best_kernels.json`: pinned winners

Pinned entries are keyed by:

- `op_name`
- `mlx_version`
- `device_arch`
- `device_name`
- `dtype`
- `shape_signature`

## Runtime Use

Runtime loading is opt-in:

```bash
export ZMLX_USE_DISCOVERED_KERNELS=1
export ZMLX_DISCOVERED_KERNELS_PATH=configs/discovered_kernels.json  # optional
```

Current runtime hooks:

- `zmlx.patch._modules.ZMLXRMSNorm` (`rmsnorm` lookup)
- `zmlx.kernels.transformer.swiglu2` (`swiglu` lookup)
- `zmlx.kernels.rope.rope_concat_qk_decode_pos` (`rope` lookup)

If no matching entry exists for the exact `(MLX version, device arch/name, dtype, shape)`, ZMLX falls back to the existing implementation.

## Benchmark Correctness Notes

The evaluator uses:

- Warmup iterations
- Timed iterations
- Forced boundaries with `mx.eval(...)` and `mx.synchronize()`

Per-candidate metrics include:

- `latency_us` (median)
- `p10_us` / `p90_us`
- `speedup_vs_ref`
- `gbps_est`
- `correctness_max_abs_err` / `correctness_max_rel_err`
