"""ZMLX-specific training callbacks."""

from __future__ import annotations

from typing import Any


class KernelStatsCallback:
    """Logs ZMLX kernel statistics during training.

    Prints cache utilization and kernel call counts at specified intervals.
    """

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.step = 0

    def on_step_end(self, step: int, loss: float, **kwargs: Any) -> None:
        self.step = step
        if step > 0 and step % self.log_interval == 0:
            self._report()

    def _report(self) -> None:
        from zmlx.registry import cache_size, list_kernels

        print(f"  [zmlx] Step {self.step}: {cache_size()} cached kernels")
        kernels = list_kernels()
        if kernels:
            print(f"  [zmlx] Active kernels: {', '.join(kernels[:10])}")
            if len(kernels) > 10:
                print(f"  [zmlx] ... and {len(kernels) - 10} more")


class PatchSummaryCallback:
    """Prints the patch summary at training start."""

    def on_train_start(self, model: Any, **kwargs: Any) -> None:
        result = getattr(model, "_zmlx_patch_result", None)
        if result:
            print(f"[zmlx] Model patched: {result.summary()}")
        else:
            print("[zmlx] Model not patched (no ZMLX kernel acceleration)")
