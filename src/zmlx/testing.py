"""Testing utilities for custom Metal kernels.

Provides helpers to verify custom kernels against reference implementations,
including both value correctness and gradient checks.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ._compat import import_mx


def assert_matches(
    kernel_fn: Callable[..., Any],
    ref_fn: Callable[..., Any],
    shapes: Sequence[tuple[int, ...]],
    *,
    dtypes: Sequence[Any] | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    n_inputs: int = 1,
    seed: int = 42,
) -> None:
    """Assert that a custom kernel matches a reference function across shapes.

    Args:
        kernel_fn: The custom kernel callable to test.
        ref_fn: Reference implementation (e.g. an MLX built-in).
        shapes: Sequence of input shapes to test.
        dtypes: Sequence of dtypes to test. Defaults to ``[mx.float32]``.
        atol: Absolute tolerance for comparison.
        rtol: Relative tolerance for comparison.
        n_inputs: Number of input arrays to generate per test case.
        seed: Random seed for reproducibility.

    Raises:
        AssertionError: If any test case fails the tolerance check.
    """
    mx = import_mx()
    if dtypes is None:
        dtypes = [mx.float32]

    mx.random.seed(seed)

    for shape in shapes:
        for dtype in dtypes:
            inputs = [mx.random.normal(shape).astype(dtype) for _ in range(n_inputs)]

            got = kernel_fn(*inputs)
            expected = ref_fn(*inputs)

            mx.eval(got, expected)

            if got.shape != expected.shape:
                raise AssertionError(
                    f"Shape mismatch for shape={shape}, dtype={dtype}: "
                    f"got {got.shape}, expected {expected.shape}"
                )

            diff = mx.abs(got.astype(mx.float32) - expected.astype(mx.float32))
            ref_abs = mx.abs(expected.astype(mx.float32))
            within_tol = diff <= (atol + rtol * ref_abs)
            mx.eval(within_tol)

            if not mx.all(within_tol).item():
                max_diff = mx.max(diff).item()
                raise AssertionError(
                    f"Value mismatch for shape={shape}, dtype={dtype}: "
                    f"max abs diff={max_diff:.6e} (atol={atol}, rtol={rtol})"
                )


def assert_gradient_matches(
    kernel_fn: Callable[..., Any],
    ref_fn: Callable[..., Any],
    shapes: Sequence[tuple[int, ...]],
    *,
    dtypes: Sequence[Any] | None = None,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    seed: int = 42,
) -> None:
    """Assert that the gradient of a custom kernel matches a reference gradient.

    Both ``kernel_fn`` and ``ref_fn`` must be differentiable (support ``mx.grad``).
    The test wraps each in a ``sum()`` loss to obtain a scalar for differentiation.

    Args:
        kernel_fn: The custom kernel callable to test.
        ref_fn: Reference implementation.
        shapes: Sequence of input shapes to test.
        dtypes: Sequence of dtypes to test. Defaults to ``[mx.float32]``.
        atol: Absolute tolerance for gradient comparison.
        rtol: Relative tolerance for gradient comparison.
        seed: Random seed for reproducibility.

    Raises:
        AssertionError: If any gradient test case fails the tolerance check.
    """
    mx = import_mx()
    if dtypes is None:
        dtypes = [mx.float32]

    mx.random.seed(seed)

    def _make_loss(fn: Callable) -> Callable:
        def loss(x: Any) -> Any:
            return fn(x).sum()
        return loss

    kernel_loss = _make_loss(kernel_fn)
    ref_loss = _make_loss(ref_fn)

    kernel_grad = mx.grad(kernel_loss)
    ref_grad = mx.grad(ref_loss)

    for shape in shapes:
        for dtype in dtypes:
            x = mx.random.normal(shape).astype(dtype)

            got = kernel_grad(x)
            expected = ref_grad(x)

            mx.eval(got, expected)

            if got.shape != expected.shape:
                raise AssertionError(
                    f"Gradient shape mismatch for shape={shape}, dtype={dtype}: "
                    f"got {got.shape}, expected {expected.shape}"
                )

            diff = mx.abs(got.astype(mx.float32) - expected.astype(mx.float32))
            ref_abs = mx.abs(expected.astype(mx.float32))
            within_tol = diff <= (atol + rtol * ref_abs)
            mx.eval(within_tol)

            if not mx.all(within_tol).item():
                max_diff = mx.max(diff).item()
                raise AssertionError(
                    f"Gradient mismatch for shape={shape}, dtype={dtype}: "
                    f"max abs diff={max_diff:.6e} (atol={atol}, rtol={rtol})"
                )


__all__ = [
    "assert_matches",
    "assert_gradient_matches",
]
