"""Expression tree for Metal code generation.

Expressions are data, not strings. Each node is a frozen dataclass that
supports Python operator overloading and lowers to Metal C via ``to_metal()``.

The ``RawMetal`` escape hatch preserves exact hand-written strings for
incremental migration without byte-identity breakage.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Expr:
    """Base class for expression tree nodes.

    Supports arithmetic operators that return new tree nodes.
    """

    def to_metal(self) -> str:
        raise NotImplementedError(f"{type(self).__name__}.to_metal()")

    # Arithmetic — returns tree nodes, not strings
    def __add__(self, other: Expr | float | int) -> BinOp:
        return BinOp("+", self, _coerce(other))

    def __radd__(self, other: float | int) -> BinOp:
        return BinOp("+", _coerce(other), self)

    def __sub__(self, other: Expr | float | int) -> BinOp:
        return BinOp("-", self, _coerce(other))

    def __rsub__(self, other: float | int) -> BinOp:
        return BinOp("-", _coerce(other), self)

    def __mul__(self, other: Expr | float | int) -> BinOp:
        return BinOp("*", self, _coerce(other))

    def __rmul__(self, other: float | int) -> BinOp:
        return BinOp("*", _coerce(other), self)

    def __truediv__(self, other: Expr | float | int) -> BinOp:
        return BinOp("/", self, _coerce(other))

    def __rtruediv__(self, other: float | int) -> BinOp:
        return BinOp("/", _coerce(other), self)

    def __neg__(self) -> UnaryOp:
        return UnaryOp("-", self)

    def __gt__(self, other: Expr | float | int) -> BinOp:
        return BinOp(">", self, _coerce(other))

    def __lt__(self, other: Expr | float | int) -> BinOp:
        return BinOp("<", self, _coerce(other))

    def __ge__(self, other: Expr | float | int) -> BinOp:
        return BinOp(">=", self, _coerce(other))

    def __le__(self, other: Expr | float | int) -> BinOp:
        return BinOp("<=", self, _coerce(other))


@dataclass(frozen=True)
class Var(Expr):
    """A named variable reference.  ``Var("x")`` → ``"x"``."""

    name: str

    def to_metal(self) -> str:
        return self.name


@dataclass(frozen=True)
class Const(Expr):
    """A typed constant.  ``Const(1.0)`` → ``"(T)1.0"``."""

    value: float | int

    def to_metal(self) -> str:
        v = self.value
        if isinstance(v, float):
            # Ensure .0 suffix for round floats
            s = repr(v)
            if "." not in s and "e" not in s and "E" not in s:
                s = s + ".0"
            return f"(T){s}"
        return f"(T){v}"


@dataclass(frozen=True)
class RawMetal(Expr):
    """Escape hatch: emit an exact Metal string.

    Use during migration to preserve byte-identical output.
    """

    code: str

    def to_metal(self) -> str:
        return self.code


@dataclass(frozen=True)
class BinOp(Expr):
    """Binary operation.  ``BinOp("+", lhs, rhs)`` → ``"(lhs + rhs)"``."""

    op: str
    lhs: Expr
    rhs: Expr

    def to_metal(self) -> str:
        return f"({self.lhs.to_metal()} {self.op} {self.rhs.to_metal()})"


@dataclass(frozen=True)
class UnaryOp(Expr):
    """Unary operation.  ``UnaryOp("-", x)`` → ``"(-(x))"``."""

    op: str
    operand: Expr

    def to_metal(self) -> str:
        return f"({self.op}({self.operand.to_metal()}))"


@dataclass(frozen=True)
class Call(Expr):
    """Function call.  ``Call("metal::exp", [x])`` → ``"metal::exp(x)"``."""

    func: str
    args: tuple[Expr, ...]

    def to_metal(self) -> str:
        arg_strs = ", ".join(a.to_metal() for a in self.args)
        return f"{self.func}({arg_strs})"


@dataclass(frozen=True)
class Let(Expr):
    """Local binding (the ``vjp_prelude`` replacement).

    ``Let("s", sigmoid(x), body_using_s)`` generates::

        T s = <init.to_metal()>;
        <body.to_metal()>  // can reference s
    """

    var_name: str
    init: Expr
    body: Expr

    def to_metal(self) -> str:
        # Let is structural — when used as a top-level prelude, the caller
        # extracts the binding. As an expression, we emit the body (the
        # binding must be hoisted by the caller).
        return self.body.to_metal()

    def to_metal_prelude(self) -> str:
        """Emit the ``T var = init;`` statement."""
        return f"T {self.var_name} = {self.init.to_metal()};"


@dataclass(frozen=True)
class Ternary(Expr):
    """Ternary conditional.

    ``Ternary(cond, a, b)`` → ``"(cond) ? (a) : (b)"``
    """

    cond: Expr
    if_true: Expr
    if_false: Expr

    def to_metal(self) -> str:
        return (
            f"({self.cond.to_metal()}) ? "
            f"({self.if_true.to_metal()}) : "
            f"({self.if_false.to_metal()})"
        )


# ---------------------------------------------------------------------------
# Coercion helper
# ---------------------------------------------------------------------------

ExprLike = Expr | float | int


def _coerce(v: ExprLike) -> Expr:
    """Coerce a Python scalar to a Const node."""
    if isinstance(v, Expr):
        return v
    if isinstance(v, (int, float)):
        return Const(v)
    raise TypeError(f"Cannot coerce {type(v).__name__} to Expr")  # pragma: no cover


# ---------------------------------------------------------------------------
# Built-in constructors (return Call nodes, not strings)
# ---------------------------------------------------------------------------

def exp(x: Expr) -> Call:
    return Call("metal::exp", (x,))


def log(x: Expr) -> Call:
    return Call("metal::log", (x,))


def tanh(x: Expr) -> Call:
    return Call("metal::tanh", (x,))


def sigmoid(x: Expr) -> Call:
    return Call("kk_sigmoid", (x,))


def silu(x: Expr) -> Call:
    return Call("kk_silu", (x,))


def gelu_tanh(x: Expr) -> Call:
    return Call("kk_gelu_tanh", (x,))


def sqrt(x: Expr) -> Call:
    return Call("metal::sqrt", (x,))


def rsqrt(x: Expr) -> Call:
    return Call("metal::rsqrt", (x,))


def abs_(x: Expr) -> Call:
    return Call("metal::abs", (x,))


def max_(a: Expr, b: Expr) -> Call:
    return Call("metal::max", (a, b))


def min_(a: Expr, b: Expr) -> Call:
    return Call("metal::min", (a, b))


# ---------------------------------------------------------------------------
# Convenience: collect Let bindings from an expression tree
# ---------------------------------------------------------------------------

def collect_lets(expr: Expr) -> tuple[list[Let], Expr]:
    """Walk the tree top-down, collecting nested Let nodes.

    Returns (bindings, inner_body) where bindings is a list of Let nodes
    and inner_body is the expression with all Lets stripped.
    """
    bindings: list[Let] = []
    current = expr
    while isinstance(current, Let):
        bindings.append(current)
        current = current.body
    return bindings, current
