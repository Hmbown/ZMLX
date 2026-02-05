"""Declarative pattern predicates for the patch system.

Replaces ad-hoc ``_is_*_block()`` functions with composable predicate
combinators that implement the ``PatchPattern`` protocol.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Predicate base and combinators
# ---------------------------------------------------------------------------

class Predicate:
    """Base class for composable module predicates.

    Supports ``&`` (and), ``|`` (or), ``~`` (not) operators.
    """

    def __call__(self, module: Any, name: str, parent: Any | None = None) -> bool:
        raise NotImplementedError

    def __and__(self, other: Predicate) -> And:
        return And(self, other)

    def __or__(self, other: Predicate) -> Or:
        return Or(self, other)

    def __invert__(self) -> Not:
        return Not(self)


@dataclass(frozen=True)
class And(Predicate):
    left: Predicate
    right: Predicate

    def __call__(self, module: Any, name: str, parent: Any | None = None) -> bool:
        return self.left(module, name, parent) and self.right(module, name, parent)


@dataclass(frozen=True)
class Or(Predicate):
    left: Predicate
    right: Predicate

    def __call__(self, module: Any, name: str, parent: Any | None = None) -> bool:
        return self.left(module, name, parent) or self.right(module, name, parent)


@dataclass(frozen=True)
class Not(Predicate):
    inner: Predicate

    def __call__(self, module: Any, name: str, parent: Any | None = None) -> bool:
        return not self.inner(module, name, parent)


# ---------------------------------------------------------------------------
# Concrete predicates
# ---------------------------------------------------------------------------

class _HasAttr(Predicate):
    """Match modules that have a given attribute."""

    def __init__(self, attr_name: str) -> None:
        self.attr_name = attr_name

    def __call__(self, module: Any, name: str, parent: Any | None = None) -> bool:
        return hasattr(module, self.attr_name)

    def __repr__(self) -> str:
        return f"has_attr({self.attr_name!r})"


class _IsInstance(Predicate):
    """Match modules that are instances of a given class."""

    def __init__(self, cls: type) -> None:
        self.cls = cls

    def __call__(self, module: Any, name: str, parent: Any | None = None) -> bool:
        return isinstance(module, self.cls)

    def __repr__(self) -> str:
        return f"is_instance({self.cls.__name__})"


class _ModulePathContains(Predicate):
    """Match modules whose ``__module__`` contains a substring."""

    def __init__(self, substring: str) -> None:
        self.substring = substring

    def __call__(self, module: Any, name: str, parent: Any | None = None) -> bool:
        mod_path = getattr(type(module), "__module__", "") or ""
        return self.substring in mod_path

    def __repr__(self) -> str:
        return f"module_path_contains({self.substring!r})"


class _ClassName(Predicate):
    """Match modules whose class name equals a string."""

    def __init__(self, cls_name: str) -> None:
        self.cls_name = cls_name

    def __call__(self, module: Any, name: str, parent: Any | None = None) -> bool:
        return type(module).__name__ == self.cls_name

    def __repr__(self) -> str:
        return f"class_name_is({self.cls_name!r})"


class _AttrNameMatches(Predicate):
    """Match based on the attribute name (the ``name`` arg to matches())."""

    def __init__(self, pattern: str) -> None:
        self.pattern = pattern
        self._re = re.compile(pattern)

    def __call__(self, module: Any, name: str, parent: Any | None = None) -> bool:
        return bool(self._re.search(name))

    def __repr__(self) -> str:
        return f"attr_name_matches({self.pattern!r})"


class _ParentHasAttr(Predicate):
    """Match when the parent module has a given attribute."""

    def __init__(self, attr_name: str) -> None:
        self.attr_name = attr_name

    def __call__(self, module: Any, name: str, parent: Any | None = None) -> bool:
        if parent is None:
            return False
        return hasattr(parent, self.attr_name)

    def __repr__(self) -> str:
        return f"parent_has_attr({self.attr_name!r})"


class _Custom(Predicate):
    """Match using an arbitrary callable."""

    def __init__(self, fn: Callable[..., bool]) -> None:
        self.fn = fn

    def __call__(self, module: Any, name: str, parent: Any | None = None) -> bool:
        return self.fn(module, name, parent)

    def __repr__(self) -> str:
        return f"custom({self.fn!r})"


# ---------------------------------------------------------------------------
# Public predicate constructors
# ---------------------------------------------------------------------------

def has_attr(attr_name: str) -> Predicate:
    """Module has attribute ``attr_name``."""
    return _HasAttr(attr_name)


def is_instance(cls: type) -> Predicate:
    """Module is instance of ``cls``."""
    return _IsInstance(cls)


def module_path_contains(substring: str) -> Predicate:
    """Module's ``__module__`` contains ``substring``."""
    return _ModulePathContains(substring)


def class_name_is(cls_name: str) -> Predicate:
    """Module's class ``__name__`` equals ``cls_name``."""
    return _ClassName(cls_name)


def attr_name_matches(pattern: str) -> Predicate:
    """The attribute name (in the model tree) matches a regex ``pattern``."""
    return _AttrNameMatches(pattern)


def parent_has_attr(attr_name: str) -> Predicate:
    """Parent module has attribute ``attr_name``."""
    return _ParentHasAttr(attr_name)


def custom(fn: Callable[..., bool]) -> Predicate:
    """Match using an arbitrary callable ``fn(module, name, parent) -> bool``."""
    return _Custom(fn)


# ---------------------------------------------------------------------------
# Bridge to PatchPattern protocol
# ---------------------------------------------------------------------------

class DeclarativePattern:
    """Implements the ``PatchPattern`` protocol using predicate combinators.

    Parameters
    ----------
    pattern_name:
        Name for this pattern (e.g. ``"moe_mlp"``).
    predicate:
        Composed predicate for ``matches()``.
    apply_fn:
        ``(module, config) -> replacement_module`` for ``apply()``.
    """

    def __init__(
        self,
        pattern_name: str,
        predicate: Predicate,
        apply_fn: Callable[[Any, Any], Any],
    ) -> None:
        self._name = pattern_name
        self._predicate = predicate
        self._apply_fn = apply_fn

    @property
    def name(self) -> str:
        return self._name

    def matches(self, module: Any, name: str, parent: Any | None = None) -> bool:
        return self._predicate(module, name, parent)

    def apply(self, module: Any, config: Any) -> Any:
        return self._apply_fn(module, config)
