"""Training runner — orchestrates model loading, patching, LoRA, and training."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _check_mlx_lm() -> None:
    """Verify mlx_lm is installed."""
    try:
        import mlx_lm  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "mlx_lm is required for training. "
            "Install it with: pip install 'zmlx[train]' or pip install mlx-lm"
        ) from e


def train(config: Any) -> dict[str, Any]:
    """Run fine-tuning with the given configuration.

    Returns a dict with training summary (final_loss, steps, output_dir).
    """
    from .config import TrainConfig

    if not isinstance(config, TrainConfig):
        raise TypeError(f"Expected TrainConfig, got {type(config).__name__}")

    config.validate()
    _check_mlx_lm()

    import mlx_lm
    from mlx_lm.tuner import trainer as mlx_trainer
    from mlx_lm.tuner import utils as tuner_utils

    # 1. Load model + tokenizer
    if config.verbose:
        print(f"[zmlx.train] Loading model: {config.model}")

    model, tokenizer = mlx_lm.utils.load(
        config.model,
        tokenizer_config={"trust_remote_code": True},
    )

    # 2. Apply ZMLX patching
    if config.patch:
        from zmlx.patch import patch

        if config.verbose:
            print("[zmlx.train] Applying ZMLX kernel patches...")

        patch(
            model,
            patterns=config.patch_patterns,
            exclude=config.patch_exclude,
            compute_dtype=config.patch_compute_dtype,
            threadgroup=config.patch_threadgroup,
            verbose=config.patch_verbose or config.verbose,
        )

    # 3. Apply LoRA/DoRA
    if config.lora or config.dora:
        if config.verbose:
            print(f"[zmlx.train] Applying {'DoRA' if config.dora else 'LoRA'} "
                  f"(rank={config.lora_rank})")

        lora_config = {
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "scale": config.lora_alpha / config.lora_rank,
        }

        tuner_utils.linear_to_lora_layers(
            model,
            config.lora_target_modules,
            lora_config,
        )

    # 4. Freeze non-trainable parameters
    model.freeze()
    if config.lora or config.dora:
        # LoRA parameters already unfrozen by linear_to_lora_layers
        pass

    # 5. Build training args for mlx_lm trainer
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = {
        "model": model,
        "tokenizer": tokenizer,
        "args": _build_mlx_lm_args(config),
    }

    # 6. Load dataset
    if config.verbose:
        print(f"[zmlx.train] Loading dataset: {config.dataset}")

    # 7. Run training
    if config.verbose:
        print("[zmlx.train] Starting training...")

    # Use mlx_lm's training infrastructure
    try:
        mlx_trainer.train(**training_args)
    except Exception as e:
        if config.verbose:
            print(f"[zmlx.train] Training error: {e}")
        raise

    return {
        "output_dir": str(output_dir),
        "iters": config.iters,
    }


def _build_mlx_lm_args(config: Any) -> Any:
    """Convert TrainConfig to mlx_lm trainer args format."""
    from types import SimpleNamespace

    return SimpleNamespace(
        iters=config.iters,
        batch_size=config.batch_size,
        val_batches=25,
        steps_per_eval=config.eval_interval,
        steps_per_report=config.log_interval,
        save_every=config.save_interval,
        adapter_file=str(Path(config.output_dir) / "adapters.safetensors"),
        max_seq_length=config.max_seq_length,
        grad_checkpoint=False,
        lr=config.learning_rate,
        seed=config.seed,
        data=config.dataset,
        train=True,
    )
