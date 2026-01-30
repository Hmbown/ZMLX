"""High-level UX helpers for ZMLX (Unsloth-style)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map, tree_unflatten


def _check_mlx_lm() -> None:
    try:
        import mlx_lm  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "mlx_lm is required for this API. Install with: pip install 'zmlx[train]' "
            "or pip install mlx-lm"
        ) from e


def _resolve_dtype(dtype: Any) -> Any:
    if dtype is None:
        return None
    if isinstance(dtype, str):
        if not hasattr(mx, dtype):
            raise ValueError(f"Unknown dtype: {dtype!r}")
        return getattr(mx, dtype)
    return dtype


def _cast_model_dtype(model: nn.Module, dtype: Any) -> None:
    if dtype is None:
        return

    def _cast_param(p: Any) -> Any:
        if hasattr(p, "dtype") and mx.issubdtype(p.dtype, mx.floating):
            return p.astype(dtype)
        return p

    model.update(tree_map(_cast_param, model.parameters()))


def _parse_quantize(quantize: str | int | None) -> int | None:
    if quantize is None:
        return None
    if isinstance(quantize, int):
        if quantize in (4, 8):
            return quantize
        raise ValueError("quantize must be 4 or 8 when given as int")
    q = quantize.lower().replace("bit", "")
    if q in ("4", "4b"):
        return 4
    if q in ("8", "8b"):
        return 8
    raise ValueError("quantize must be '4bit', '8bit', 4, 8, or None")


def load(
    model_name: str,
    *,
    quantize: str | None = None,
    patch: bool = True,
    patch_patterns: list[str] | None = None,
    dtype: str = "float16",
    verbose: bool = False,
) -> tuple[nn.Module, Any]:
    """Load model + tokenizer, optionally quantize and patch with ZMLX fused kernels."""
    _check_mlx_lm()
    from mlx_lm import utils as lm_utils

    if verbose:
        print(f"[zmlx.load] Loading model: {model_name}")

    model, tokenizer, config = lm_utils.load(
        model_name,
        tokenizer_config={"trust_remote_code": True},
        return_config=True,
    )

    q_bits = _parse_quantize(quantize)
    already_quantized = bool(config.get("quantization") or config.get("quantization_config"))
    if q_bits is not None and not already_quantized:
        if verbose:
            print(f"[zmlx.load] Quantizing to {q_bits}-bit (group_size=64)")
        model, _ = lm_utils.quantize_model(
            model,
            config,
            group_size=64,
            bits=q_bits,
            mode="affine",
        )
    elif q_bits is not None and already_quantized and verbose:
        print("[zmlx.load] Model already quantized; skipping re-quantization.")

    cd = _resolve_dtype(dtype)
    _cast_model_dtype(model, cd)

    if patch:
        from zmlx.patch import patch as zmlx_patch

        if verbose:
            print("[zmlx.load] Applying ZMLX patching...")
        zmlx_patch(
            model,
            patterns=patch_patterns,
            compute_dtype=dtype,
            verbose=verbose,
        )

    return model, tokenizer


def lora(
    model: nn.Module,
    *,
    r: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
    use_dora: bool = False,
) -> nn.Module:
    """Apply LoRA/DoRA adapters to a model in-place."""
    _check_mlx_lm()

    import mlx.nn as nn
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear
    from mlx_lm.tuner.dora import DoRAEmbedding, DoRALinear
    from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear, LoRASwitchLinear

    if r <= 0:
        raise ValueError("r must be > 0")

    model.freeze()

    scale = alpha / float(r)

    def _to_lora(layer: nn.Module) -> nn.Module:
        if use_dora:
            if isinstance(layer, (nn.Linear, nn.QuantizedLinear)):
                return DoRALinear.from_base(layer, r=r, dropout=dropout, scale=scale)
            if isinstance(layer, (nn.Embedding, nn.QuantizedEmbedding)):
                return DoRAEmbedding.from_base(layer, r=r, dropout=dropout, scale=scale)
            raise ValueError(f"DoRA unsupported for layer type: {type(layer).__name__}")

        if hasattr(layer, "to_lora"):
            return layer.to_lora(r=r, dropout=dropout, scale=scale)
        if isinstance(layer, (nn.Linear, nn.QuantizedLinear)):
            return LoRALinear.from_base(layer, r=r, dropout=dropout, scale=scale)
        if isinstance(layer, (SwitchLinear, QuantizedSwitchLinear)):
            return LoRASwitchLinear.from_base(layer, r=r, dropout=dropout, scale=scale)
        if isinstance(layer, (nn.Embedding, nn.QuantizedEmbedding)):
            return LoRAEmbedding.from_base(layer, r=r, dropout=dropout, scale=scale)
        raise ValueError(f"Unsupported layer type for LoRA: {type(layer).__name__}")

    lora_layers = []
    target_suffixes = tuple(target_modules) if target_modules else None

    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRASwitchLinear, LoRAEmbedding, DoRALinear, DoRAEmbedding)):
            continue
        if target_suffixes is not None and not name.endswith(target_suffixes):
            continue

        if hasattr(module, "to_lora") or isinstance(
            module,
            (
                nn.Linear,
                nn.QuantizedLinear,
                SwitchLinear,
                QuantizedSwitchLinear,
                nn.Embedding,
                nn.QuantizedEmbedding,
            ),
        ):
            lora_layers.append((name, _to_lora(module)))

    if not lora_layers:
        raise ValueError("No matching modules found for LoRA application.")

    model.update_modules(tree_unflatten(lora_layers))
    return model


def train(
    model: nn.Module,
    tokenizer: Any,
    dataset: str | dict,
    *,
    iters: int = 1000,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    output_dir: str = "adapters",
    **kwargs,
) -> dict:
    """Train with sensible defaults. Returns training stats."""
    _check_mlx_lm()

    import mlx.optimizers as optim
    from mlx_lm.tuner import datasets as tuner_datasets
    from mlx_lm.tuner import trainer
    from mlx_lm.tuner.utils import build_schedule

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    use_fused_loss = bool(kwargs.pop("use_fused_loss", True))
    max_seq_length = int(kwargs.pop("max_seq_length", 2048))
    eval_interval = int(kwargs.pop("eval_interval", 200))
    log_interval = int(kwargs.pop("log_interval", 10))
    save_interval = int(kwargs.pop("save_interval", 100))
    val_batches = int(kwargs.pop("val_batches", 25))
    grad_accum_steps = int(kwargs.pop("grad_accum_steps", 1))
    grad_checkpoint = bool(kwargs.pop("grad_checkpoint", False))
    mask_prompt = bool(kwargs.pop("mask_prompt", False))

    optimizer_name = str(kwargs.pop("optimizer", "adam")).lower()
    optimizer_config = dict(kwargs.pop("optimizer_config", {}))
    lr_schedule = kwargs.pop("lr_schedule", None)

    args = SimpleNamespace(
        data=dataset if isinstance(dataset, str) else "",
        hf_dataset=dataset if isinstance(dataset, dict) else None,
        train=True,
        test=False,
        mask_prompt=mask_prompt,
        max_seq_length=max_seq_length,
    )

    train_set, valid_set, _ = tuner_datasets.load_dataset(args, tokenizer)

    adapter_file = output_path / "adapters.safetensors"

    training_args = trainer.TrainingArgs(
        batch_size=batch_size,
        iters=iters,
        val_batches=val_batches,
        steps_per_report=log_interval,
        steps_per_eval=eval_interval,
        steps_per_save=save_interval,
        max_seq_length=max_seq_length,
        adapter_file=str(adapter_file),
        grad_checkpoint=grad_checkpoint,
        grad_accumulation_steps=grad_accum_steps,
    )

    lr = build_schedule(lr_schedule) if isinstance(lr_schedule, dict) else learning_rate

    if optimizer_name == "adam":
        opt_class = optim.Adam
    elif optimizer_name == "adamw":
        opt_class = optim.AdamW
    elif optimizer_name == "muon":
        opt_class = optim.Muon
    elif optimizer_name == "sgd":
        opt_class = optim.SGD
    elif optimizer_name == "adafactor":
        opt_class = optim.Adafactor
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    optimizer = opt_class(learning_rate=lr, **optimizer_config)

    if use_fused_loss:
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

    trainer.train(
        model=model,
        optimizer=optimizer,
        train_dataset=tuner_datasets.CacheDataset(train_set),
        val_dataset=tuner_datasets.CacheDataset(valid_set),
        args=training_args,
        loss=loss_fn,
    )

    return {
        "output_dir": str(output_path),
        "adapter_file": str(adapter_file),
        "iters": iters,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }


def generate(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    max_tokens: int = 256,
    temp: float = 0.7,
) -> str:
    """Generate text from a patched model."""
    _check_mlx_lm()
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=float(temp))
    return mlx_lm.generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    )


__all__ = [
    "load",
    "lora",
    "train",
    "generate",
]
