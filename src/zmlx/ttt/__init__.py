"""TTT (Test-Time Training) layers for MLX.

Implements TTT-Linear: a sequence modeling layer that replaces attention
with a learned weight matrix updated via gradient descent at inference time.
The "hidden state" is model weights rather than a fixed-size vector.

Reference: Sun et al., "Learning to (Learn at Test Time): RNNs with
Expressive Hidden States", arXiv:2407.04620.
"""

from __future__ import annotations

from .linear import TTTCache, TTTLinear, TTTLinearConfig

__all__ = ["TTTLinear", "TTTLinearConfig", "TTTCache"]
