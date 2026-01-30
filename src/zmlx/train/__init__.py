"""zmlx.train — Fine-tuning suite for MLX models.

Thin wrapper over ``mlx_lm`` that adds ZMLX kernel acceleration
via ``zmlx.patch()`` and fused loss functions.

Usage::

    from zmlx.train import train, TrainConfig

    config = TrainConfig(model="mlx-community/Llama-3.2-1B", lora=True)
    train(config)

CLI::

    zmlx train --model mlx-community/Llama-3.2-1B --lora --dataset alpaca
"""

from __future__ import annotations

from .config import TrainConfig
from .runner import train

__all__ = ["TrainConfig", "train"]
