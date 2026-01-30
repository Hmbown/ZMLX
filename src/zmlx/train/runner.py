"""Training runner — orchestrates model loading, patching, LoRA, and training."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
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

    import mlx.core as mx
    import mlx.optimizers as optim
    import mlx_lm
    from mlx_lm.tuner import datasets as tuner_datasets
    from mlx_lm.tuner import trainer
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

    # 3. Freeze base parameters, then apply LoRA/DoRA (so new LoRA params stay unfrozen)
    model.freeze()

    if config.lora or config.dora:
        if config.verbose:
            print(
                f"[zmlx.train] Applying {'DoRA' if config.dora else 'LoRA'} "
                f"(rank={config.lora_rank})"
            )

        num_layers = len(model.layers) if hasattr(model, "layers") else 0

        # Expand leaf target names (e.g. "q_proj") to full dotted paths
        # (e.g. "self_attn.q_proj") as required by linear_to_lora_layers.
        targets = config.lora_target_modules
        if targets and hasattr(model, "layers") and model.layers:
            full_keys: set[str] = set()
            for k, _m in model.layers[0].named_modules():
                if any(k == t or k.endswith(f".{t}") for t in targets):
                    full_keys.add(k)
        else:
            full_keys = None  # type: ignore[assignment]

        lora_config: dict[str, Any] = {
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "scale": config.lora_alpha / config.lora_rank,
        }
        if full_keys:
            lora_config["keys"] = full_keys

        tuner_utils.linear_to_lora_layers(
            model,
            num_layers,
            lora_config,
            use_dora=config.dora,
        )

    if config.verbose:
        tuner_utils.print_trainable_parameters(model)

    # 5. Build output dir
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = output_dir / "adapters.safetensors"

    # 6. Load dataset
    if config.verbose:
        print(f"[zmlx.train] Loading dataset: {config.dataset}")

    dataset_args = SimpleNamespace(
        data=config.dataset,
        hf_dataset=None,
        train=True,
        test=False,
        mask_prompt=False,
        max_seq_length=config.max_seq_length,
    )

    train_set, valid_set, _ = tuner_datasets.load_dataset(dataset_args, tokenizer)

    # 7. Build TrainingArgs
    training_args = trainer.TrainingArgs(
        batch_size=config.batch_size,
        iters=config.iters,
        val_batches=25,
        steps_per_report=config.log_interval,
        steps_per_eval=config.eval_interval,
        steps_per_save=config.save_interval,
        max_seq_length=config.max_seq_length,
        adapter_file=str(adapter_file),
        grad_checkpoint=False,
        grad_accumulation_steps=config.grad_accum_steps,
    )

    # 8. Build optimizer
    optimizer_name = config.optimizer.lower()
    lr = config.learning_rate
    if optimizer_name == "adam":
        optimizer = optim.Adam(learning_rate=lr)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(learning_rate=lr)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(learning_rate=lr)
    elif optimizer_name == "adafactor":
        optimizer = optim.Adafactor(learning_rate=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # 9. Build loss function
    if config.use_fused_loss:
        from zmlx.train.loss import fused_cross_entropy

        def loss_fn(model_in, batch, lengths):
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits = model_in(inputs)
            steps = mx.arange(1, targets.shape[1] + 1)
            mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
            ce = fused_cross_entropy(logits, targets, reduction="none") * mask
            ntoks = mask.sum()
            ce = ce.astype(mx.float32).sum() / ntoks
            return ce, ntoks

    else:
        loss_fn = trainer.default_loss

    # 10. Build callback
    training_callback = None
    if config.use_callbacks:
        from .callbacks import ZMLXCallback

        training_callback = ZMLXCallback(
            model=model,
            log_interval=config.log_interval,
            verbose=config.verbose,
        )

    # 11. Run training
    if config.verbose:
        print("[zmlx.train] Starting training...")

    trainer.train(
        model=model,
        optimizer=optimizer,
        train_dataset=tuner_datasets.CacheDataset(train_set),
        val_dataset=tuner_datasets.CacheDataset(valid_set),
        args=training_args,
        loss=loss_fn,
        training_callback=training_callback,
    )

    return {
        "output_dir": str(output_dir),
        "adapter_file": str(adapter_file),
        "iters": config.iters,
    }
