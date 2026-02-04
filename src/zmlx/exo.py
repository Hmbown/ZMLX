"""zmlx.exo — Zero-setup exo integration via runtime monkey-patch.

Launch exo with ZMLX fused-kernel patching, no source modifications needed::

    pip install zmlx exo
    python -m zmlx.exo

How it works:

1. ``install_hook()`` wraps ``exo.worker.engines.mlx.utils_mlx.load_mlx_items``
   so that every model load triggers ``zmlx.patch.patch(model, ...)``.
2. ``main()`` sets env vars, prepends a bootstrap directory to ``PYTHONPATH``
   (so spawned subprocesses also get the hook via ``sitecustomize.py``), then
   calls ``exo.main.main()``.

Configuration env vars (same as the git-patch integration):

- ``EXO_ZMLX=1``           — enable patching (set automatically by ``main()``)
- ``EXO_ZMLX_VERBOSE=1``   — print per-module patch log
- ``EXO_ZMLX_PATTERNS=...`` — comma-separated pattern list override
- ``EXO_ZMLX_EXCLUDE=...``  — comma-separated patterns to skip
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import time
from typing import Any

logger = logging.getLogger("zmlx.exo")

_hook_installed = False


def _apply_zmlx_patch(model: Any, bound_instance: Any, group: Any) -> Any:
    """Apply ZMLX fused-kernel patches to a loaded exo model."""
    if os.environ.get("EXO_ZMLX", "").strip().lower() not in {"1", "true", "yes"}:
        return model

    try:
        from zmlx.patch import patch as zmlx_patch
    except Exception:
        logger.warning("[zmlx.exo] Failed to import zmlx.patch — continuing unpatched")
        return model

    verbose = os.environ.get("EXO_ZMLX_VERBOSE", "").strip().lower() in {
        "1", "true", "yes",
    }

    # Resolve patterns from env
    patterns_env = os.environ.get("EXO_ZMLX_PATTERNS", "").strip()
    exclude_env = os.environ.get("EXO_ZMLX_EXCLUDE", "").strip()

    patch_kwargs: dict[str, Any] = {"verbose": verbose}
    exclude_patterns: list[str] = []

    if patterns_env:
        patch_kwargs["patterns"] = [
            p.strip() for p in patterns_env.split(",") if p.strip()
        ]
    else:
        # Tensor-parallel: skip moe_mlp — exo's ShardedMoE wrappers replace
        # the MoE __call__ with distributed all-reduce logic.  Dense SwiGLU
        # layers within experts still benefit from swiglu_mlp fusion.
        if bound_instance is not None:
            shard = getattr(bound_instance, "bound_shard", None)
            if shard is not None and "Tensor" in type(shard).__name__:
                exclude_patterns.append("moe_mlp")
                logger.info(
                    "[zmlx.exo] Tensor-parallel mode: excluding moe_mlp "
                    "(exo handles MoE distribution; dense SwiGLU still fused)"
                )

    if exclude_env:
        exclude_patterns.extend(
            p.strip() for p in exclude_env.split(",") if p.strip()
        )
    if exclude_patterns:
        patch_kwargs["exclude"] = sorted(set(exclude_patterns))

    is_distributed = group is not None
    parallelism = "none"
    if is_distributed and bound_instance is not None:
        shard = getattr(bound_instance, "bound_shard", None)
        if shard is not None and "Tensor" in type(shard).__name__:
            parallelism = "tensor"
        else:
            parallelism = "pipeline"

    logger.info(
        f"[zmlx.exo] Applying fused-kernel patches "
        f"(distributed={is_distributed}, parallelism={parallelism})..."
    )
    t0 = time.perf_counter()

    try:
        model = zmlx_patch(model, **patch_kwargs)
    except Exception:
        logger.exception("[zmlx.exo] Patching failed — continuing with unpatched model")
        return model

    elapsed = time.perf_counter() - t0
    result = getattr(model, "_zmlx_patch_result", None)
    if result is not None and result.patched_count > 0:
        logger.info(
            f"[zmlx.exo] Patched {result.patched_count} modules in {elapsed:.2f}s: "
            f"{dict(result.pattern_counts)}"
        )
    elif result is not None:
        logger.info(
            f"[zmlx.exo] No modules matched in {elapsed:.2f}s "
            "(model may use non-standard naming or all patterns were excluded)"
        )
    else:
        logger.info(f"[zmlx.exo] Patching completed in {elapsed:.2f}s")

    return model


def install_hook() -> None:
    """Monkey-patch exo's ``load_mlx_items`` to apply ZMLX after model load."""
    global _hook_installed
    if _hook_installed:
        return

    try:
        utils_mlx = importlib.import_module("exo.worker.engines.mlx.utils_mlx")
    except ModuleNotFoundError:
        logger.debug("[zmlx.exo] exo.worker.engines.mlx.utils_mlx not found — skipping hook")
        return

    original_load = getattr(utils_mlx, "load_mlx_items", None)
    if original_load is None:
        logger.warning("[zmlx.exo] load_mlx_items not found in utils_mlx — skipping hook")
        return

    # Don't double-wrap
    if getattr(original_load, "_zmlx_wrapped", False):
        _hook_installed = True
        return

    def patched_load_mlx_items(
        bound_instance: Any,
        group: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        result = original_load(bound_instance, group, *args, **kwargs)
        # load_mlx_items returns a tuple (model, tokenizer, ...)
        if isinstance(result, tuple) and len(result) >= 1:
            model = _apply_zmlx_patch(result[0], bound_instance, group)
            result = (model, *result[1:])
        return result

    patched_load_mlx_items._zmlx_wrapped = True  # type: ignore[attr-defined]
    utils_mlx.load_mlx_items = patched_load_mlx_items  # type: ignore[attr-defined]
    _hook_installed = True
    logger.info("[zmlx.exo] Hook installed on load_mlx_items")


def main() -> None:
    """Entry point: set up env, install hook, launch exo."""
    # Enable ZMLX patching
    os.environ["EXO_ZMLX"] = "1"
    os.environ.setdefault("EXO_ZMLX_VERBOSE", "1")
    os.environ["_ZMLX_EXO_HOOK"] = "1"

    # Prepend bootstrap dir to PYTHONPATH so spawned subprocesses get
    # sitecustomize.py (which calls install_hook() on interpreter startup).
    bootstrap_dir = os.path.join(os.path.dirname(__file__), "_exo_bootstrap")
    pythonpath = os.environ.get("PYTHONPATH", "")
    if bootstrap_dir not in pythonpath:
        os.environ["PYTHONPATH"] = (
            bootstrap_dir + os.pathsep + pythonpath if pythonpath else bootstrap_dir
        )

    # Install hook in the parent process
    install_hook()

    print("[zmlx.exo] Starting exo with ZMLX fused-kernel patching...")

    # Import and run exo
    try:
        from exo.main import main as exo_main
    except ModuleNotFoundError:
        print(
            "Error: exo is not installed. Install with:\n"
            "  pip install exo\n"
            "or:\n"
            "  pip install git+https://github.com/exo-explore/exo.git"
        )
        sys.exit(1)

    exo_main()


if __name__ == "__main__":
    main()
