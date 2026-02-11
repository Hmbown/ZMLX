# Foundry Module

The foundry generates Metal kernel variant datasets for evaluation and training data export. It replaces four earlier prototype repos (Discover, Lab, Foundry, DataFoundry) with a single module under `src/zmlx/foundry/`.

## What it does

1. **Op registry** -- 16 registered ops across 9 kernel classes (rmsnorm, layernorm, softmax, swiglu, geglu, moe_combine, moe_gating, rope, dropout).
2. **Metal templates** -- discovers `.metal` template files with knob placeholders (`{{BLOCK_SIZE}}`, `{{USE_FMA}}`, etc.) and renders concrete variants.
3. **Harness** -- compiles each variant, runs correctness checks against the MLX reference op, and benchmarks throughput.
4. **Sampling** -- random, coverage-guided, mutation, and CEM-based candidate generation strategies.
5. **Curriculum scheduler** -- staged op unlocking across difficulty tiers.
6. **Export** -- NDJSON training data and chat-style SFT JSONL for kernel-writing fine-tuning.

## CLI

```bash
python -m zmlx.foundry run                  # generate kernel dataset
python -m zmlx.foundry run --ops rmsnorm swiglu --n 500 --workers 4
python -m zmlx.foundry report sessions/my_run   # coverage + pareto reports
python -m zmlx.foundry export sessions/my_run --out training_data/
python -m zmlx.foundry export-sft --out training_data/kernel_sft
python -m zmlx.foundry list                  # list registered ops
```

## SFT export workflow

The `export-sft` command produces chat-format JSONL suitable for `mlx_lm.lora` fine-tuning. It ingests from four sources: foundry sessions, KD runs, discover sessions, and raw templates. Output includes train/valid/test splits with deduplication and a manifest JSON.

Example end-to-end training:

```bash
# 1. Export SFT dataset
python -m zmlx.foundry export-sft --out training_data/kernel_sft

# 2. Run LoRA fine-tuning
python -m mlx_lm.lora \
  --model mlx-community/Qwen3-1.7B-4bit \
  --data training_data/kernel_sft \
  --train \
  --config configs/qwen3_1p7b_kernel_sft_lora.yaml
```

See `configs/qwen3_1p7b_kernel_sft_lora.yaml` for the training configuration.

## Module layout

```
src/zmlx/foundry/
  __init__.py         # module docstring, __all__
  __main__.py         # CLI (run, report, export, export-sft, list)
  taxonomy.py         # foundation types (KernelClass, OpSpec, KernelCandidate)
  ids.py              # stable identifiers for attempts and cache keys
  ndjson.py           # append-only NDJSON logging
  scheduler.py        # curriculum-based staged op unlocking
  session.py          # session directory management
  workers.py          # multi-process worker orchestration
  ops/                # op registry (16 ops)
  templates/          # Metal template discovery and rendering
  harness/            # compile, correctness, benchmark orchestrator
  sampling/           # candidate generation strategies
  plugins/            # protocol-based extensibility
  reports/            # coverage analysis and Pareto extraction
  export/             # training JSONL + chat SFT export
```

## Adding a new op

1. Create a template `.metal` file in `templates/`.
2. Register the op in `ops/` with its `OpSpec` (kernel class, reference function, shapes, dtypes).
3. Run `python -m zmlx.foundry run --ops your_op --n 100` to verify the harness evaluates it correctly.
4. Run `python -m zmlx.foundry report` to check coverage.
