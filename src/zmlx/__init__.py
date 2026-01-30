"""ZMLX

Ergonomic helpers around MLX custom Metal kernels.

Core API:
- metal.kernel: build/get a cached kernel wrapper
- elementwise.unary / elementwise.binary: quickly create elementwise ops
- autograd.unary_from_expr / autograd.binary_from_expr: ops with custom VJP using Metal kernels
- autotune.autotune_threadgroup: simple threadgroup search

Extras:
- kernels.*: a catalog of ready-to-use fused kernels (softmax, norms, RoPE, activations, ...)
- codegen/msl: small building blocks for generating Metal source strings
"""

__version__ = "0.2.1"

from ._compat import is_supported_host

_IMPORT_ERROR = (
    "ZMLX requires macOS on Apple Silicon (M-series) for Metal kernels. "
    "This import is blocked on unsupported platforms."
)

def __getattr__(name: str):
    if not is_supported_host():
        raise RuntimeError(_IMPORT_ERROR)

    if name in {"autograd", "elementwise", "kernels", "metal", "registry", "rowwise", "msl"}:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")

    if name == "autotune_threadgroup":
        from .autotune import autotune_threadgroup as _autotune_threadgroup

        return _autotune_threadgroup

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)

__all__ = [
    "__version__",
    "autograd",
    "elementwise",
    "kernels",
    "metal",
    "msl",
    "autotune_threadgroup",
    "registry",
    "rowwise",
]
