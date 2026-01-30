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


def _get_training_callback_base() -> type:
    """Import TrainingCallback base class from mlx_lm, with fallback."""
    try:
        from mlx_lm.tuner.callbacks import TrainingCallback

        return TrainingCallback  # type: ignore[no-any-return]
    except ImportError:
        return object


class ZMLXCallback(_get_training_callback_base()):  # type: ignore[misc]
    """Adapter that wraps ZMLX callbacks into mlx_lm's TrainingCallback interface.

    Bridges the gap between ZMLX's ``KernelStatsCallback`` /
    ``PatchSummaryCallback`` and the ``TrainingCallback`` protocol used by
    ``mlx_lm.tuner.trainer.train()``.
    """

    def __init__(
        self,
        model: Any,
        log_interval: int = 100,
        verbose: bool = False,
    ):
        self._kernel_stats = KernelStatsCallback(log_interval=log_interval)
        self._patch_summary = PatchSummaryCallback()
        self._model = model
        self._verbose = verbose
        self._started = False

    def on_train_loss_report(self, train_info: dict) -> None:
        if not self._started:
            self._patch_summary.on_train_start(model=self._model)
            self._started = True

        iteration = train_info.get("iteration", 0)
        train_loss = train_info.get("train_loss", 0.0)
        self._kernel_stats.on_step_end(step=iteration, loss=train_loss)

    def on_val_loss_report(self, val_info: dict) -> None:
        if not self._started:
            self._patch_summary.on_train_start(model=self._model)
            self._started = True
