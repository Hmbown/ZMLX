"""zmlx.vllm_metal — Zero-setup vllm-metal integration via runtime monkey-patch.

Enable ZMLX fused-kernel patching for vllm-metal with no source modifications::

    # Option 1: env var (with vllm-metal already installed)
    ZMLX_VLLM=1 vllm serve <model>

    # Option 2: explicit in code
    import zmlx.vllm_metal
    zmlx.vllm_metal.enable()

    # Option 3: auto-detected via vLLM general_plugins entry point
    pip install zmlx  # alongside vllm-metal; set ZMLX_VLLM=1

How it works:

1. ``enable()`` wraps ``vllm_metal.v1.model_runner.MetalModelRunner.load_model``
   so that every model load triggers ``zmlx.patch.patch(model, ...)``.
2. The same wrap is applied to the legacy model runner at
   ``vllm_metal.model_runner.MetalModelRunner.load_model``.
3. When registered as a ``vllm.general_plugins`` entry point, the
   ``register()`` function calls ``enable()`` automatically if ``ZMLX_VLLM=1``.

Configuration env vars:

- ``ZMLX_VLLM=1``            — enable patching (required for auto-plugin)
- ``ZMLX_VLLM_VERBOSE=1``    — print per-module patch log
- ``ZMLX_VLLM_PATTERNS=...`` — comma-separated pattern list override
- ``ZMLX_VLLM_EXCLUDE=...``  — comma-separated patterns to skip
- ``ZMLX_VLLM_MODE=...``     — patch mode: "inference" (default) or "training"
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

logger = logging.getLogger("zmlx.vllm_metal")

_enabled = False
_TRUTHY = {"1", "true", "yes"}
_v1_original_load_model: Callable[..., Any] | None = None
_legacy_original_load_model: Callable[..., Any] | None = None
_v1_runner_cls: type[Any] | None = None
_legacy_runner_cls: type[Any] | None = None


def _get_env_list(key: str) -> list[str] | None:
    """Parse a comma-separated env var into a list, or None if unset."""
    val = os.environ.get(key, "").strip()
    if not val:
        return None
    return [s.strip() for s in val.split(",") if s.strip()]


def _env_true(key: str) -> bool:
    """Return True when env var is set to a truthy value."""
    return os.environ.get(key, "").strip().lower() in _TRUTHY


def _patch_model(model: Any, model_name: str = "") -> Any:
    """Apply ZMLX fused kernels to an MLX model.

    Args:
        model: The MLX model to patch.
        model_name: Model name for logging.

    Returns:
        The patched model (modified in place).
    """
    try:
        from zmlx.patch import patch as zmlx_patch
    except ImportError:
        logger.warning(
            "zmlx.patch not available — skipping ZMLX patching. "
            "Is ZMLX installed? (`pip install zmlx`)"
        )
        return model

    verbose = _env_true("ZMLX_VLLM_VERBOSE")
    patterns = _get_env_list("ZMLX_VLLM_PATTERNS")
    exclude = _get_env_list("ZMLX_VLLM_EXCLUDE")
    mode = os.environ.get("ZMLX_VLLM_MODE", "inference").strip()

    kwargs: dict[str, Any] = {"verbose": verbose}
    if patterns is not None:
        kwargs["patterns"] = patterns
    else:
        kwargs["mode"] = mode
    if exclude is not None:
        kwargs["exclude"] = exclude

    t0 = time.perf_counter()
    try:
        model = zmlx_patch(model, **kwargs)
        dt = time.perf_counter() - t0

        result = getattr(model, "_zmlx_patch_result", None)
        n_patched = int(getattr(result, "patched_count", 0) or 0)
        logger.info(
            "ZMLX patched %s: %d modules replaced in %.3fs",
            model_name or "model",
            n_patched,
            dt,
        )
    except Exception:
        logger.exception(
            "ZMLX patching failed for %s — continuing with unpatched model",
            model_name or "model",
        )

    return model


def _wrap_v1_model_runner() -> bool:
    """Monkey-patch the v1 MetalModelRunner.load_model to inject ZMLX.

    Returns:
        True if the patch was applied, False if vllm-metal v1 is not available.
    """
    global _v1_original_load_model, _v1_runner_cls

    try:
        from vllm_metal.v1 import model_runner as v1_mr
    except ImportError:
        return False

    Runner = v1_mr.MetalModelRunner
    if getattr(Runner.load_model, "_zmlx_wrapped", False):
        _v1_runner_cls = Runner
        return True  # already wrapped

    _original = Runner.load_model
    _v1_original_load_model = _original
    _v1_runner_cls = Runner

    @wraps(_original)
    def _load_model_with_zmlx(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = _original(self, *args, **kwargs)
        if not _enabled:
            return result
        model_name = ""
        if hasattr(self, "model_config") and self.model_config is not None:
            mc = self.model_config
            model_name = getattr(mc, "model", "") if hasattr(mc, "model") else ""
        self.model = _patch_model(self.model, model_name)
        return result

    _load_model_with_zmlx._zmlx_wrapped = True  # type: ignore[attr-defined]
    _load_model_with_zmlx._zmlx_original = _original  # type: ignore[attr-defined]
    Runner.load_model = _load_model_with_zmlx  # type: ignore[assignment]
    logger.debug("Wrapped vllm_metal.v1.model_runner.MetalModelRunner.load_model")
    return True


def _wrap_legacy_model_runner() -> bool:
    """Monkey-patch the legacy MetalModelRunner.load_model to inject ZMLX.

    Returns:
        True if the patch was applied, False if not available.
    """
    global _legacy_original_load_model, _legacy_runner_cls

    try:
        from vllm_metal import model_runner as legacy_mr
    except ImportError:
        return False

    Runner = legacy_mr.MetalModelRunner
    if getattr(Runner.load_model, "_zmlx_wrapped", False):
        _legacy_runner_cls = Runner
        return True

    _original = Runner.load_model
    _legacy_original_load_model = _original
    _legacy_runner_cls = Runner

    @wraps(_original)
    def _load_model_with_zmlx(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = _original(self, *args, **kwargs)
        if not _enabled:
            return result
        model_name = ""
        if hasattr(self, "vllm_config"):
            mc = getattr(self.vllm_config, "model_config", None)
            if mc is not None:
                model_name = getattr(mc, "model", "")
        self.model = _patch_model(self.model, model_name)
        return result

    _load_model_with_zmlx._zmlx_wrapped = True  # type: ignore[attr-defined]
    _load_model_with_zmlx._zmlx_original = _original  # type: ignore[attr-defined]
    Runner.load_model = _load_model_with_zmlx  # type: ignore[assignment]
    logger.debug("Wrapped vllm_metal.model_runner.MetalModelRunner.load_model")
    return True


def enable() -> None:
    """Enable ZMLX optimizations for vllm-metal.

    Wraps vllm-metal's model runner(s) so that every model load
    automatically applies ZMLX fused Metal kernels.

    Safe to call multiple times (idempotent).

    Raises:
        ImportError: If vllm-metal is not installed.

    Example::

        import zmlx.vllm_metal
        zmlx.vllm_metal.enable()
        # Now start vllm as normal — models will be ZMLX-patched on load.
    """
    global _enabled
    if _enabled:
        return

    v1_ok = _wrap_v1_model_runner()
    legacy_ok = _wrap_legacy_model_runner()

    if not v1_ok and not legacy_ok:
        raise ImportError(
            "vllm-metal is not installed or not importable. "
            "Install it with: pip install vllm-metal"
        )

    _enabled = True
    runners = []
    if v1_ok:
        runners.append("v1")
    if legacy_ok:
        runners.append("legacy")
    logger.info("ZMLX vllm-metal integration enabled (runners: %s)", ", ".join(runners))


def disable() -> None:
    """Disable ZMLX optimizations for vllm-metal.

    Restores the original load_model methods. Only affects future
    model loads — already-patched models remain patched.
    """
    global _enabled, _legacy_original_load_model, _legacy_runner_cls
    global _v1_original_load_model, _v1_runner_cls

    if not _enabled and _v1_runner_cls is None and _legacy_runner_cls is None:
        return

    # Stop patching first, then restore original methods if we have them.
    _enabled = False

    if _v1_runner_cls is not None and _v1_original_load_model is not None:
        if getattr(_v1_runner_cls.load_model, "_zmlx_wrapped", False):
            _v1_runner_cls.load_model = _v1_original_load_model  # type: ignore[assignment]
    if _legacy_runner_cls is not None and _legacy_original_load_model is not None:
        if getattr(_legacy_runner_cls.load_model, "_zmlx_wrapped", False):
            _legacy_runner_cls.load_model = _legacy_original_load_model  # type: ignore[assignment]

    _v1_original_load_model = None
    _legacy_original_load_model = None
    _v1_runner_cls = None
    _legacy_runner_cls = None
    logger.info("ZMLX vllm-metal integration disabled")


def register() -> None:
    """Entry point for vLLM's general_plugins system.

    Called automatically by vLLM at startup if ZMLX is installed and
    the entry point is registered. Only activates if ``ZMLX_VLLM=1``.
    """
    if not _env_true("ZMLX_VLLM"):
        logger.debug(
            "ZMLX vLLM plugin loaded but ZMLX_VLLM is not set — "
            "set ZMLX_VLLM=1 to enable"
        )
        return

    try:
        enable()
    except ImportError:
        logger.debug(
            "ZMLX vLLM plugin: vllm-metal not available, skipping"
        )
    except Exception:
        logger.exception("ZMLX vLLM plugin: unexpected error during enable()")
