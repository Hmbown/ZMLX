#!/usr/bin/env python3
"""Benchmark: ZMLX-patched vs baseline training on Qwen3-1.7B-4bit.

Runs two training sessions with identical hyperparameters:
  1. Baseline — standard mlx_lm training (no patches, standard CE loss)
  2. ZMLX    — ZMLX kernel patches + fused cross-entropy loss

Captures tokens/sec, peak memory, and loss curves, then saves results as JSON.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.optimizers as optim
import mlx_lm
from mlx.utils import tree_flatten
from mlx_lm.tuner import datasets as tuner_datasets
from mlx_lm.tuner import trainer
from mlx_lm.tuner import utils as tuner_utils
from mlx_lm.tuner.callbacks import TrainingCallback

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL = "mlx-community/Qwen3-1.7B-4bit"
DATASET = "mlx-community/WikiSQL"
ITERS = 100
BATCH_SIZE = 2
LR = 2e-4
MAX_SEQ = 512
LORA_RANK = 8
LORA_ALPHA = 16.0
LORA_TARGETS = ["q_proj", "v_proj", "k_proj", "o_proj"]
LOG_EVERY = 10
EVAL_EVERY = 50
OUTPUT_BASE = Path("benchmarks/results")


# ---------------------------------------------------------------------------
# Metric-collecting callback
# ---------------------------------------------------------------------------
@dataclass
class MetricCollector(TrainingCallback):
    """Collects training metrics from mlx_lm trainer callbacks."""

    train_losses: list = field(default_factory=list)
    val_losses: list = field(default_factory=list)
    tokens_per_sec: list = field(default_factory=list)
    iters_per_sec: list = field(default_factory=list)
    peak_memory_gb: list = field(default_factory=list)

    def on_train_loss_report(self, info: dict) -> None:
        self.train_losses.append(
            {"iteration": info["iteration"], "loss": info["train_loss"]}
        )
        self.tokens_per_sec.append(info.get("tokens_per_second", 0))
        self.iters_per_sec.append(info.get("iterations_per_second", 0))
        self.peak_memory_gb.append(info.get("peak_memory", 0))

    def on_val_loss_report(self, info: dict) -> None:
        self.val_losses.append(
            {"iteration": info["iteration"], "loss": info["val_loss"]}
        )

    def summary(self) -> dict:
        tps = self.tokens_per_sec
        ips = self.iters_per_sec
        mem = self.peak_memory_gb
        losses = [e["loss"] for e in self.train_losses]
        return {
            "avg_tokens_per_sec": sum(tps) / len(tps) if tps else 0,
            "avg_iters_per_sec": sum(ips) / len(ips) if ips else 0,
            "peak_memory_gb": max(mem) if mem else 0,
            "final_train_loss": losses[-1] if losses else None,
            "first_train_loss": losses[0] if losses else None,
            "train_loss_curve": self.train_losses,
            "val_loss_curve": self.val_losses,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_model():
    """Load model + tokenizer from HF, cast to float16 for kernel compat."""
    from mlx.utils import tree_map

    model, tokenizer = mlx_lm.utils.load(
        MODEL, tokenizer_config={"trust_remote_code": True}
    )

    # Cast to float16 (matching zmlx.load default) for Metal kernel compatibility
    def _cast(p):
        if hasattr(p, "dtype") and mx.issubdtype(p.dtype, mx.floating):
            return p.astype(mx.float16)
        return p

    model.update(tree_map(_cast, model.parameters()))
    return model, tokenizer


def _apply_lora(model):
    """Freeze base weights, then apply LoRA (so new LoRA params stay unfrozen)."""
    model.freeze()
    num_layers = len(model.layers) if hasattr(model, "layers") else 0

    # Expand leaf target names to full dotted paths for linear_to_lora_layers
    full_keys = set()
    if hasattr(model, "layers") and model.layers:
        for k, _m in model.layers[0].named_modules():
            if any(k == t or k.endswith(f".{t}") for t in LORA_TARGETS):
                full_keys.add(k)

    lora_config = {
        "rank": LORA_RANK,
        "alpha": LORA_ALPHA,
        "dropout": 0.0,
        "scale": LORA_ALPHA / LORA_RANK,
    }
    if full_keys:
        lora_config["keys"] = full_keys

    tuner_utils.linear_to_lora_layers(model, num_layers, lora_config)
    return model


def _load_dataset(tokenizer):
    """Load WikiSQL dataset."""
    args = SimpleNamespace(
        data=DATASET,
        hf_dataset=None,
        train=True,
        test=False,
        mask_prompt=False,
        max_seq_length=MAX_SEQ,
    )
    train_set, valid_set, _ = tuner_datasets.load_dataset(args, tokenizer)
    return (
        tuner_datasets.CacheDataset(train_set),
        tuner_datasets.CacheDataset(valid_set),
    )


def _training_args(output_dir: Path) -> trainer.TrainingArgs:
    output_dir.mkdir(parents=True, exist_ok=True)
    return trainer.TrainingArgs(
        batch_size=BATCH_SIZE,
        iters=ITERS,
        val_batches=25,
        steps_per_report=LOG_EVERY,
        steps_per_eval=EVAL_EVERY,
        steps_per_save=ITERS,  # save only at end
        max_seq_length=MAX_SEQ,
        adapter_file=str(output_dir / "adapters.safetensors"),
        grad_checkpoint=False,
    )


def _print_trainable(model):
    total = sum(p.size for _, p in tree_flatten(model.parameters())) / 1e6
    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters())) / 1e6
    print(f"  Parameters: {trainable:.1f}M trainable / {total:.1f}M total "
          f"({trainable / total * 100:.2f}%)")


# ---------------------------------------------------------------------------
# Run one training session
# ---------------------------------------------------------------------------
def run_training(
    label: str,
    use_patches: bool,
    use_fused_loss: bool,
    output_dir: Path,
) -> dict:
    """Run a single training session and return metrics."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  patches={use_patches}  fused_loss={use_fused_loss}")
    print(f"{'='*60}\n")

    # Fresh model load
    print(f"[{label}] Loading model: {MODEL}")
    model, tokenizer = _load_model()

    # Optional ZMLX patching
    if use_patches:
        from zmlx.patch import patch as zmlx_patch

        print(f"[{label}] Applying ZMLX patches...")
        zmlx_patch(
            model,
            compute_dtype="float16",
            verbose=True,
        )

    # LoRA
    print(f"[{label}] Applying LoRA (rank={LORA_RANK})")
    model = _apply_lora(model)
    _print_trainable(model)

    # Dataset (reuse tokenizer)
    print(f"[{label}] Loading dataset: {DATASET}")
    train_ds, val_ds = _load_dataset(tokenizer)

    # Training args
    args = _training_args(output_dir)

    # Optimizer
    optimizer = optim.Adam(learning_rate=LR)

    # Loss
    if use_fused_loss:
        from zmlx.train.loss import fused_cross_entropy

        def loss_fn(mdl, batch, lengths):
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits = mdl(inputs)
            steps = mx.arange(1, targets.shape[1] + 1)
            mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
            ce = fused_cross_entropy(logits, targets, reduction="none") * mask
            ntoks = mask.sum()
            return ce.astype(mx.float32).sum() / ntoks, ntoks

    else:
        loss_fn = trainer.default_loss

    # Callback
    collector = MetricCollector()

    # Train
    wall_start = time.perf_counter()
    trainer.train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        args=args,
        loss=loss_fn,
        training_callback=collector,
    )
    wall_time = time.perf_counter() - wall_start

    summary = collector.summary()
    summary["wall_time_sec"] = round(wall_time, 2)
    summary["label"] = label
    summary["use_patches"] = use_patches
    summary["use_fused_loss"] = use_fused_loss

    print(f"\n[{label}] Done in {wall_time:.1f}s")
    print(f"  Avg tokens/sec: {summary['avg_tokens_per_sec']:.1f}")
    print(f"  Peak memory:    {summary['peak_memory_gb']:.2f} GB")
    print(f"  Final loss:     {summary['final_train_loss']}")

    # Cleanup to free memory before next run
    del model, optimizer, train_ds, val_ds
    gc.collect()
    if hasattr(mx, "clear_memory_cache"):
        mx.clear_memory_cache()

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Run 1: Baseline
    baseline = run_training(
        label="Baseline (no patches)",
        use_patches=False,
        use_fused_loss=False,
        output_dir=OUTPUT_BASE / "baseline",
    )

    # Run 2: ZMLX
    zmlx_result = run_training(
        label="ZMLX (patches + fused loss)",
        use_patches=True,
        use_fused_loss=True,
        output_dir=OUTPUT_BASE / "zmlx",
    )

    # Comparison
    print("\n" + "=" * 60)
    print("  BENCHMARK RESULTS")
    print("=" * 60)

    speedup = (
        zmlx_result["avg_tokens_per_sec"] / baseline["avg_tokens_per_sec"]
        if baseline["avg_tokens_per_sec"] > 0
        else 0
    )
    mem_diff = zmlx_result["peak_memory_gb"] - baseline["peak_memory_gb"]

    print(f"\n{'Metric':<30} {'Baseline':>15} {'ZMLX':>15} {'Delta':>15}")
    print("-" * 75)
    print(
        f"{'Tokens/sec':<30} "
        f"{baseline['avg_tokens_per_sec']:>15.1f} "
        f"{zmlx_result['avg_tokens_per_sec']:>15.1f} "
        f"{speedup:>14.2f}x"
    )
    print(
        f"{'Wall time (sec)':<30} "
        f"{baseline['wall_time_sec']:>15.1f} "
        f"{zmlx_result['wall_time_sec']:>15.1f} "
        f"{zmlx_result['wall_time_sec'] - baseline['wall_time_sec']:>+15.1f}"
    )
    print(
        f"{'Peak memory (GB)':<30} "
        f"{baseline['peak_memory_gb']:>15.2f} "
        f"{zmlx_result['peak_memory_gb']:>15.2f} "
        f"{mem_diff:>+15.2f}"
    )
    print(
        f"{'Final train loss':<30} "
        f"{baseline['final_train_loss']:>15.4f} "
        f"{zmlx_result['final_train_loss']:>15.4f} "
        f"{zmlx_result['final_train_loss'] - baseline['final_train_loss']:>+15.4f}"
    )

    # Save results
    results = {
        "model": MODEL,
        "dataset": DATASET,
        "config": {
            "iters": ITERS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "max_seq_length": MAX_SEQ,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_targets": LORA_TARGETS,
        },
        "baseline": baseline,
        "zmlx": zmlx_result,
        "comparison": {
            "speedup_x": round(speedup, 3),
            "memory_delta_gb": round(mem_diff, 3),
            "wall_time_delta_sec": round(
                zmlx_result["wall_time_sec"] - baseline["wall_time_sec"], 2
            ),
        },
    }

    results_path = OUTPUT_BASE / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
