from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.optimizers as optim

from .kernels.optimizers import adamw_step


class AdamW(optim.AdamW):
    """Fused AdamW optimizer.
    
    This optimizer uses a single Metal kernel to update m, v, and parameters,
    reducing memory bandwidth usage compared to standard MLX AdamW.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_lr = self.learning_rate

    def apply_gradients(self, gradients: dict, model: Any):
        if "step" not in self.state:
            self.state["step"] = mx.array(1, dtype=mx.int32)
        else:
            self.state["step"] += 1

        lr = self.learning_rate
        if callable(lr):
            lr = lr(self.state["step"])
        self._current_lr = lr

        return super().apply_gradients(gradients, model)

    def _update_single(self, p: Any, g: Any, state: dict[str, Any]):
        # This is called by super().apply_gradients -> super().update -> _update_single
        # To avoid re-creating arrays every time, we can cache them or use them from self.state
        
        if "_step_arr" not in self.state:
            self.state["_step_arr"] = mx.array([self.state["step"]], dtype=mx.float32)
        if "_lr_arr" not in self.state:
            self.state["_lr_arr"] = mx.array([self._current_lr], dtype=mx.float32)
            
        # Update them if they changed (step always changes)
        self.state["_step_arr"] = mx.array([self.state["step"]], dtype=mx.float32)
        self.state["_lr_arr"] = mx.array([self._current_lr], dtype=mx.float32)

        if "m" not in state:
            state["m"] = mx.zeros_like(p)
        if "v" not in state:
            state["v"] = mx.zeros_like(p)
            
        new_p, new_m, new_v = adamw_step(
            p, g, state["m"], state["v"],
            self.state["_lr_arr"], self.state["_step_arr"],
            beta1=self.betas[0],
            beta2=self.betas[1],
            eps=self.eps,
            wd=self.weight_decay
        )
        
        state["m"] = new_m
        state["v"] = new_v
        return new_p
