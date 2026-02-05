"""ZMLX Kernel DSL — declarative kernel definitions.

Public API re-exports for the DSL package.
"""

from __future__ import annotations

from .defkernel import defkernel, defkernel_mapreduce
from .expr import (
    BinOp,
    Call,
    Const,
    Expr,
    Let,
    RawMetal,
    Ternary,
    UnaryOp,
    Var,
    abs_,
    collect_lets,
    exp,
    gelu_tanh,
    log,
    max_,
    min_,
    rsqrt,
    sigmoid,
    silu,
    sqrt,
    tanh,
)
from .family import Axis, KernelFamily
from .match import (
    DeclarativePattern,
    Predicate,
    attr_name_matches,
    class_name_is,
    custom,
    has_attr,
    is_instance,
    module_path_contains,
    parent_has_attr,
)

__all__ = [
    # Expression tree
    "Expr",
    "Var",
    "Const",
    "RawMetal",
    "BinOp",
    "UnaryOp",
    "Call",
    "Let",
    "Ternary",
    "collect_lets",
    # Built-in math
    "exp",
    "log",
    "tanh",
    "sigmoid",
    "silu",
    "gelu_tanh",
    "sqrt",
    "rsqrt",
    "abs_",
    "max_",
    "min_",
    # Kernel definition
    "defkernel",
    "defkernel_mapreduce",
    # Family
    "Axis",
    "KernelFamily",
    # Pattern matching
    "Predicate",
    "DeclarativePattern",
    "has_attr",
    "is_instance",
    "module_path_contains",
    "class_name_is",
    "attr_name_matches",
    "parent_has_attr",
    "custom",
]
