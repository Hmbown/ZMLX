"""High-level API for ZMLX — kernel authoring and inference helpers.

Kernel authoring (primary API):
    - ``elementwise(expr, ...)`` — custom elementwise op with automatic gradient
    - ``reduce(init, update, finalize, ...)`` — custom rowwise reduction
    - ``map_reduce(pass1, pass2, write, ...)`` — two-pass rowwise map-reduce
    - ``jit(fn)`` — decorator for JIT-compiling Python scalar ops to Metal

Model helpers (requires mlx-lm):
    - ``load()``, ``generate()``
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from .jit_compiler import jit


def elementwise(
    expr: str,
    *,
    name: str | None = None,
    grad_expr: str | None = None,
    compute_dtype: Any = None,
    header: str | None = None,
    use_output: bool = True,
    grad_prelude: str = "",
) -> Callable[[Any], Any]:
    """Create a custom elementwise op from a C expression.

    This is the simplest path to a tested, differentiable Metal kernel.

    Args:
        expr: Metal/C expression for the forward pass, in terms of ``x``
            (the input value cast to compute dtype).
        name: Kernel name for caching and debugging. Defaults to a name
            derived from the expression.
        grad_expr: Metal expression for the VJP backward pass.  Available
            variables: ``g`` (upstream gradient), ``y`` (forward output if
            ``use_output=True``), ``x`` (input if ``use_output=False``).
            If ``None``, the returned op is not differentiable.
        compute_dtype: MLX dtype for the ``T`` template parameter.
            Defaults to ``mx.float32``.
        header: Optional Metal header. Defaults to the built-in snippet
            library (sigmoid, silu, gelu_tanh helpers).
        use_output: Whether the backward kernel uses the forward output
            (``y``) or the original input (``x``). Default ``True``.
        grad_prelude: Optional Metal statements inserted before the gradient
            expression (e.g. ``"T s = kk_sigmoid(x);"``).

    Returns:
        A callable that takes a single MLX array and returns an MLX array.
        If ``grad_expr`` is provided, the callable supports ``mx.grad``.

    Example::

        import zmlx

        mish = zmlx.elementwise(
            "x * tanh(log(1 + exp(x)))",
            name="mish",
            grad_expr="...",  # supply VJP expression for differentiability
        )
        y = mish(mx.random.normal((1024,)))
    """
    from .msl import DEFAULT_HEADER

    cd = compute_dtype or mx.float32
    hdr = header if header is not None else DEFAULT_HEADER
    kernel_name = name or f"zmlx_ew_{abs(hash(expr)) % 100000}"

    if grad_expr is not None:
        from .autograd import unary_from_expr

        return unary_from_expr(
            name=kernel_name,
            fwd_expr=expr,
            vjp_expr=grad_expr,
            compute_dtype=cd,
            use_output=use_output,
            header=hdr,
            vjp_prelude=grad_prelude,
        )

    from .elementwise import unary

    return unary(
        name=kernel_name,
        expr=expr,
        compute_dtype=cd,
        header=hdr,
    )


def reduce(
    *,
    init: str,
    update: str,
    finalize: str = "acc",
    name: str | None = None,
    compute_dtype: Any = None,
    header: str | None = None,
) -> Callable[[Any], Any]:
    """Create a custom rowwise reduction kernel.

    The kernel reduces each row (last dimension) of the input to a scalar.

    Args:
        init: C expression for the initial accumulator value
            (e.g. ``"0.0f"`` or ``"-INFINITY"``).
        update: C expression updating the accumulator. Available
            variables: ``acc`` (accumulator), ``v`` (current element).
        finalize: C expression to transform the final accumulator into
            the output scalar. Available variable: ``s`` (reduced value).
            Default ``"acc"`` (identity).
        name: Kernel name for caching. Auto-generated if ``None``.
        compute_dtype: MLX dtype for the ``T`` template. Default ``mx.float32``.
        header: Optional Metal header. Defaults to built-in snippet library.

    Returns:
        A callable that takes an MLX array and returns an array with the last
        dimension reduced.

    Example::

        import zmlx

        my_sum = zmlx.reduce(
            init="0.0f",
            update="acc + v",
            name="simple_sum",
        )
        y = my_sum(mx.random.normal((8, 1024)))  # shape (8,)
    """
    from .codegen import rowwise_reduction_source
    from .metal import kernel as metal_kernel
    from .msl import DEFAULT_HEADER

    cd = compute_dtype or mx.float32
    hdr = header if header is not None else DEFAULT_HEADER
    kernel_name = name or f"zmlx_reduce_{abs(hash(update)) % 100000}"

    def _make_op(d: int) -> Any:
        src = rowwise_reduction_source(
            reduce_expr=update,
            init_expr=init,
            finalize_expr=finalize,
            d=d,
        )
        return metal_kernel(
            name=f"{kernel_name}_d{d}",
            input_names=["inp"],
            output_names=["out"],
            source=src,
            header=hdr,
            ensure_row_contiguous=True,
            cache=True,
        )

    # Cache compiled kernels per last-dim size
    _cache: dict[int, Any] = {}

    def op(x: Any) -> Any:
        d = int(x.shape[-1])
        if d not in _cache:
            _cache[d] = _make_op(d)
        k = _cache[d]
        rows = x.size // d
        out_shape = x.shape[:-1] if x.ndim > 1 else (1,)
        return k(
            x,
            template=[("T", cd)],
            grid=(rows, 1, 1),
            threadgroup=(1, 1, 1),
            output_shapes=[out_shape],
            output_dtypes=[x.dtype],
        )[0]

    return op


def map_reduce(
    *,
    pass1: dict[str, str],
    pass2: dict[str, str],
    write: str,
    name: str | None = None,
    threadgroup: int | str = 256,
    compute_dtype: Any = None,
    header: str | None = None,
) -> Callable[[Any], Any]:
    """Create a two-pass rowwise map-reduce kernel.

    This is the pattern behind softmax, layer-norm, and similar ops that need
    two reductions before writing output.

    Args:
        pass1: Dict with keys ``"init"``, ``"update"``, ``"reduce"`` for the
            first reduction pass. ``update`` has vars ``acc1``, ``x``.
            ``reduce`` has vars ``a``, ``b``.
        pass2: Dict with keys ``"init"``, ``"update"``, ``"reduce"`` for the
            second reduction pass. ``update`` has vars ``acc2``, ``x``, ``s1``
            (result of pass 1). ``reduce`` has vars ``a``, ``b``.
        write: Per-element output expression. Available vars: ``x``, ``s1``
            (pass 1 result), ``s2`` (pass 2 result).
        name: Kernel name for caching. Auto-generated if ``None``.
        threadgroup: Threads per threadgroup (power of 2), or "auto" to autotune.
            Default 256.
        compute_dtype: MLX dtype for the ``T`` template. Default ``mx.float32``.
        header: Optional Metal header. Defaults to built-in snippet library.

    Returns:
        A callable that takes an MLX array and returns an array of the same
        shape.

    Example::

        import zmlx

        my_softmax = zmlx.map_reduce(
            pass1={"init": "-INFINITY", "update": "max(acc1, x)", "reduce": "max(a, b)"},
            pass2={"init": "0.0f", "update": "acc2 + exp(x - s1)", "reduce": "a + b"},
            write="exp(x - s1) / s2",
            name="my_softmax",
        )
        y = my_softmax(mx.random.normal((8, 1024)))
    """
    from .autotune import get_autotuned_config
    from .codegen import rowwise_mapreduce_source
    from .metal import kernel as metal_kernel
    from .msl import DEFAULT_HEADER

    cd = compute_dtype or mx.float32
    hdr = header if header is not None else DEFAULT_HEADER
    kernel_name = name or f"zmlx_mr_{abs(hash(write)) % 100000}"

    def _make_op(d: int, tg: int) -> Any:
        src = rowwise_mapreduce_source(
            d=d,
            tg=tg,
            pass1_init=pass1["init"],
            pass1_update=pass1["update"],
            pass1_reduce_op=pass1["reduce"],
            pass2_init=pass2["init"],
            pass2_update=pass2["update"],
            pass2_reduce_op=pass2["reduce"],
            write_expr=write,
        )
        return metal_kernel(
            name=f"{kernel_name}_d{d}_tg{tg}",
            input_names=["inp"],
            output_names=["out"],
            source=src,
            header=hdr,
            ensure_row_contiguous=True,
            cache=True,
        )

    _cache: dict[tuple[int, int], Any] = {}

    def op(x: Any) -> Any:
        d = int(x.shape[-1])
        rows = x.size // d

        if threadgroup == "auto":
            # We need a representative kernel to tune.
            # Use 256 as a representative tg for the source.
            base_tg = 256
            if (d, base_tg) not in _cache:
                _cache[(d, base_tg)] = _make_op(d, base_tg)
            
            k = _cache[(d, base_tg)]
            config = get_autotuned_config(
                k,
                inputs=[x],
                grid=lambda tg: (rows * tg[0], 1, 1),
                threadgroup_candidates=[(t, 1, 1) for t in (32, 64, 128, 256, 512, 1024)],
                output_shapes=[x.shape],
                output_dtypes=[x.dtype],
            )
            # Re-fetch or create kernel with BEST tg if different
            best_tg = config.threadgroup[0]
            if (d, best_tg) not in _cache:
                _cache[(d, best_tg)] = _make_op(d, best_tg)
            k = _cache[(d, best_tg)]
            tg = best_tg
        else:
            tg = int(threadgroup)
            if (d, tg) not in _cache:
                _cache[(d, tg)] = _make_op(d, tg)
            k = _cache[(d, tg)]

        return k(
            x,
            template=[("T", cd)],
            grid=(rows * tg, 1, 1),
            threadgroup=(tg, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
        )[0]

    return op


# ---------------------------------------------------------------------------
# Model helpers (require mlx-lm)
# ---------------------------------------------------------------------------


def _check_mlx_lm() -> None:
    try:
        import mlx_lm  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "mlx_lm is required for this API. Install with: pip install 'zmlx[lm]' "
            "or pip install mlx-lm"
        ) from e


def _resolve_dtype(dtype: Any) -> Any:
    if dtype is None:
        return None
    if isinstance(dtype, str):
        if not hasattr(mx, dtype):
            raise ValueError(f"Unknown dtype: {dtype!r}")
        return getattr(mx, dtype)
    return dtype


def _cast_model_dtype(model: nn.Module, dtype: Any) -> None:
    if dtype is None:
        return

    def _cast_param(p: Any) -> Any:
        if hasattr(p, "dtype") and mx.issubdtype(p.dtype, mx.floating):
            return p.astype(dtype)
        return p

    model.update(tree_map(_cast_param, model.parameters()))


def _parse_quantize(quantize: str | int | None) -> int | None:
    if quantize is None:
        return None
    if isinstance(quantize, int):
        if quantize in (4, 8):
            return quantize
        raise ValueError("quantize must be 4 or 8 when given as int")
    q = quantize.lower().replace("bit", "")
    if q in ("4", "4b"):
        return 4
    if q in ("8", "8b"):
        return 8
    raise ValueError("quantize must be '4bit', '8bit', 4, 8, or None")


def load(
    model_name: str,
    *,
    quantize: str | None = None,
    patch: bool = True,
    patch_patterns: list[str] | None = None,
    patch_profile: str | None = None,
    dtype: str = "float16",
    verbose: bool = False,
) -> tuple[nn.Module, Any]:
    """Load model + tokenizer, optionally quantize and patch with ZMLX fused kernels."""
    _check_mlx_lm()
    from mlx_lm import utils as lm_utils

    if verbose:
        print(f"[zmlx.load] Loading model: {model_name}")

    loaded = cast(
        tuple[Any, ...],
        lm_utils.load(
            model_name,
            tokenizer_config={"trust_remote_code": True},
            return_config=True,
        ),
    )
    model = loaded[0]
    tokenizer = loaded[1]
    config = cast(dict[str, Any], loaded[2] if len(loaded) > 2 else {})

    q_bits = _parse_quantize(quantize)
    already_quantized = bool(config.get("quantization") or config.get("quantization_config"))
    if q_bits is not None and not already_quantized:
        if verbose:
            print(f"[zmlx.load] Quantizing to {q_bits}-bit (group_size=64)")
        model, _ = lm_utils.quantize_model(
            model,
            config,
            group_size=64,
            bits=q_bits,
            mode="affine",
        )
    elif q_bits is not None and already_quantized and verbose:
        print("[zmlx.load] Model already quantized; skipping re-quantization.")

    cd = _resolve_dtype(dtype)
    _cast_model_dtype(model, cd)

    if patch:
        from zmlx.patch import patch as zmlx_patch

        if verbose:
            print("[zmlx.load] Applying ZMLX patching...")
        zmlx_patch(
            model,
            patterns=patch_patterns,
            profile=patch_profile,
            compute_dtype=dtype,
            verbose=verbose,
        )

    return model, tokenizer


def generate(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    max_tokens: int = 256,
    temp: float = 0.7,
    kv_bits: int | None = None,
    kv_group_size: int | None = None,
    quantized_kv_start: int | None = None,
) -> str:
    """Generate text from a patched model."""
    _check_mlx_lm()
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler

    from .kv_cache import kv_cache_kwargs

    sampler = make_sampler(temp=float(temp))
    kv_kwargs = kv_cache_kwargs(
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        quantized_kv_start=quantized_kv_start,
    )
    if "kv_bits" in kv_kwargs:
        from .mlx_lm_compat import (
            apply_kv_quantization_fixes,
            make_prompt_cache_for_kv_quantization,
        )

        apply_kv_quantization_fixes(
            model,
            kv_bits=int(kv_kwargs["kv_bits"]),
            verbose=False,
        )
        if "prompt_cache" not in kv_kwargs:
            try:
                kv_kwargs["prompt_cache"] = make_prompt_cache_for_kv_quantization(model)
            except Exception:
                pass
    result: str = mlx_lm.generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        **kv_kwargs,
    )
    return result


__all__ = [
    "elementwise",
    "reduce",
    "map_reduce",
    "jit",
    "load",
    "generate",
]
