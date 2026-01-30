"""ZMLX — Triton for Apple Silicon.

The ergonomic toolkit for custom Metal kernels on MLX.  Write, test, and
benchmark differentiable GPU ops with Triton-like simplicity.

Kernel authoring (from zmlx.api):
    elementwise()   — custom elementwise op (with optional gradient)
    reduce()        — custom rowwise reduction
    map_reduce()    — two-pass rowwise map-reduce (softmax, layer-norm, ...)

Testing & benchmarking:
    zmlx.testing       — assert_matches, assert_gradient_matches
    zmlx.bench         — compare() side-by-side benchmark table
    zmlx.profile       — time_kernel, memory_usage, dump_msl, kernel_stats

Building blocks:
    zmlx.metal         — MetalKernel wrapper with caching and stats
    zmlx.elementwise   — (module) lower-level unary/binary/map builders
    zmlx.autograd      — unary_from_expr, binary_from_expr, nary_from_expr
    zmlx.codegen       — MSL template generators
    zmlx.rowwise       — map_reduce builder
    zmlx.autotune      — threadgroup search and caching

Kernel catalog:
    zmlx.kernels.*     — 70+ ready-to-use and reference-implementation kernels

Model helpers (require mlx-lm):
    zmlx.load, zmlx.lora, zmlx.train, zmlx.generate
"""

__version__ = "0.6.0"

from ._compat import is_supported_host

_IMPORT_ERROR = (
    "ZMLX requires macOS on Apple Silicon (M-series) for Metal kernels. "
    "This import is blocked on unsupported platforms."
)

def __getattr__(name: str):
    if not is_supported_host():
        raise RuntimeError(_IMPORT_ERROR)

    if name == "jit":
        from .jit_compiler import jit as _jit
        return _jit

    # Submodules (includes elementwise as a module for backward compat)
    if name in {
        "autograd", "elementwise", "kernels", "metal", "registry", "rowwise", "msl",
        "patch", "codegen", "autotune", "optimizers", "nn",
        "testing", "bench", "profile",
    }:
        import importlib
        return importlib.import_module(f"{__name__}.{name}")

    # Top-level kernel authoring API (from api.py)
    if name in {"reduce", "map_reduce"}:
        from . import api as _api
        return getattr(_api, name)

    # Model helpers (from api.py)
    if name in {"load", "lora", "train", "generate"}:
        from . import api as _api
        return getattr(_api, name)

    if name == "autotune_threadgroup":
        from .autotune import autotune_threadgroup as _autotune_threadgroup
        return _autotune_threadgroup

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)

__all__ = [
    "__version__",
    # Building blocks (submodules)
    "autograd",
    "codegen",
    "elementwise",
    "metal",
    "msl",
    "rowwise",
    "autotune",
    "autotune_threadgroup",
    "optimizers",
    "jit",
    "nn",
    # Kernel authoring (from zmlx.api)
    "reduce",
    "map_reduce",
    # Testing & benchmarking
    "testing",
    "bench",
    "profile",
    # Kernel catalog
    "kernels",
    "registry",
    # Patch system
    "patch",
    # Model helpers
    "load",
    "lora",
    "train",
    "generate",
]
