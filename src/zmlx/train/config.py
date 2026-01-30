"""Training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    """Configuration for ZMLX fine-tuning.

    Combines model, data, LoRA, optimization, and ZMLX-specific settings.
    """

    # --- Model ---
    model: str = ""
    tokenizer: str | None = None
    model_dtype: str = "float16"

    # --- Data ---
    dataset: str = ""
    data_format: str = "auto"  # auto, alpaca, sharegpt, text, hf
    train_split: str = "train"
    val_split: str = "validation"
    max_seq_length: int = 2048

    # --- LoRA ---
    lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    dora: bool = False

    # --- Quantization ---
    quantize: bool = False  # Load model in quantized form
    q_bits: int = 4
    q_group_size: int = 64

    # --- Optimization ---
    optimizer: str = "adam"
    learning_rate: float = 1e-4
    lr_schedule: str = "cosine"
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # --- Training ---
    iters: int = 1000
    batch_size: int = 4
    grad_accum_steps: int = 1
    eval_interval: int = 100
    save_interval: int = 500
    output_dir: str = "adapters"
    seed: int = 42

    # --- ZMLX Patching ---
    patch: bool = True
    patch_patterns: list[str] | None = None
    patch_exclude: list[str] | None = None
    patch_compute_dtype: str = "float32"
    patch_threadgroup: int = 256
    patch_verbose: bool = False

    # --- Fused loss ---
    use_fused_loss: bool = True
    use_callbacks: bool = True

    # --- Resume ---
    resume_from: str | None = None

    # --- Logging ---
    verbose: bool = False
    log_interval: int = 10

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        """Load config from a YAML file."""
        import yaml  # type: ignore[import-untyped]

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def merge_cli(self, args: dict[str, Any]) -> None:
        """Merge CLI arguments into this config (non-None values override)."""
        for key, value in args.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)

    def validate(self) -> None:
        """Validate config consistency."""
        if not self.model:
            raise ValueError("model is required")
        if not self.dataset and not self.resume_from:
            raise ValueError("dataset is required (unless resuming)")
        if self.lora and self.dora:
            raise ValueError("Cannot use both LoRA and DoRA simultaneously")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.iters < 1:
            raise ValueError("iters must be >= 1")
