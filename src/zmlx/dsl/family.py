"""Kernel family: generate variant sets from axis combinations.

A ``KernelFamily`` takes a list of ``Axis`` objects (each with a name and
tuple of values) and a factory function, then generates the Cartesian product
of all axis values as named kernel builders.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Axis:
    """A single dimension of variation in a kernel family.

    Parameters
    ----------
    name:
        Axis name (used as kwarg to factory and namer).
    values:
        Tuple of possible values along this axis.
    """

    name: str
    values: tuple[Any, ...]


class KernelFamily:
    """Generate kernel variants from the Cartesian product of axes.

    Parameters
    ----------
    name:
        Family name (for diagnostics).
    axes:
        List of ``Axis`` objects defining the variation dimensions.
    factory:
        ``(**axis_kwargs) -> kernel_builder`` — called once per variant.
    namer:
        ``(**axis_kwargs) -> str`` — generates the variant name.
        If ``None``, uses ``name + "_" + "_".join(str(v) for v in values)``.
    """

    def __init__(
        self,
        name: str,
        axes: list[Axis],
        factory: Callable[..., Callable],
        namer: Callable[..., str] | None = None,
    ) -> None:
        self.name = name
        self.axes = axes
        self.factory = factory
        self._namer = namer
        self._variants: dict[str, Callable] | None = None

    def _default_namer(self, **kwargs: Any) -> str:
        parts = [self.name]
        for axis in self.axes:
            v = kwargs[axis.name]
            parts.append(str(v))
        return "_".join(parts)

    def _variant_name(self, **kwargs: Any) -> str:
        if self._namer is not None:
            return self._namer(**kwargs)
        return self._default_namer(**kwargs)

    def variants(self) -> dict[str, Callable]:
        """Return all variants as ``{name: builder}``."""
        if self._variants is not None:
            return self._variants

        result: dict[str, Callable] = {}
        axis_names = [a.name for a in self.axes]
        axis_values = [a.values for a in self.axes]

        for combo in itertools.product(*axis_values):
            kwargs = dict(zip(axis_names, combo, strict=True))
            variant_name = self._variant_name(**kwargs)
            result[variant_name] = self.factory(**kwargs)

        self._variants = result
        return result

    def get(self, **kwargs: Any) -> Callable:
        """Get a specific variant by axis values."""
        variant_name = self._variant_name(**kwargs)
        variants = self.variants()
        if variant_name not in variants:
            raise KeyError(
                f"No variant {variant_name!r} in family {self.name!r}. "
                f"Available: {sorted(variants.keys())}"
            )
        return variants[variant_name]

    def __len__(self) -> int:
        return len(self.variants())

    def __iter__(self):
        return iter(self.variants().items())
