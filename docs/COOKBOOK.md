# ZMLX Cookbook

Recipes for common custom kernel patterns on Apple Silicon.

## Custom Activation with Gradient

A differentiable activation function using `zmlx.api.elementwise()`.

```python
from zmlx.api import elementwise
from zmlx import msl
import mlx.core as mx

# SiLU (Swish) with analytic gradient
silu = elementwise(
    "kk_silu(x)",
    name="my_silu",
    grad_expr="g * (s + x * s * ((T)1 - s))",
    grad_prelude="T s = kk_sigmoid(x);",
    use_output=False,
    header=msl.DEFAULT_HEADER,
)

x = mx.random.normal((8, 1024))
y = silu(x)
gx = mx.grad(lambda z: silu(z).sum())(x)
mx.eval(y, gx)
```

Key points:
- `grad_expr` defines the VJP. `g` is the upstream gradient, `x` is the input.
- `grad_prelude` lets you precompute shared values (here, `s = sigmoid(x)`).
- `use_output=False` means the backward kernel reads the input `x`, not the output `y`.
- `header=msl.DEFAULT_HEADER` provides `kk_sigmoid`, `kk_silu`, `kk_gelu_tanh`.

## Custom Loss Function

A fused loss that avoids materializing intermediate arrays.

```python
from zmlx.api import reduce
import mlx.core as mx

# Entropy: -sum(x * log(x + eps))
entropy = reduce(
    init="0.0f",
    update="acc + (-v * log(v + 1e-8f))",
    name="entropy",
)

probs = mx.softmax(mx.random.normal((4, 1024)), axis=-1)
h = entropy(probs)  # shape (4,)
mx.eval(h)
```

## Custom Reduction

A simple rowwise max reduction.

```python
from zmlx.api import reduce
import mlx.core as mx

row_max = reduce(
    init="-INFINITY",
    update="max(acc, v)",
    name="row_max",
)

x = mx.random.normal((8, 2048))
maxvals = row_max(x)  # shape (8,)
mx.eval(maxvals)
```

## Softmax (Map-Reduce Pattern)

Two-pass map-reduce: first find row max, then compute exp sum, then normalize.

```python
from zmlx.api import map_reduce
import mlx.core as mx

my_softmax = map_reduce(
    pass1={"init": "-INFINITY", "update": "max(acc1, x)", "reduce": "max(a, b)"},
    pass2={"init": "0.0f", "update": "acc2 + exp(x - s1)", "reduce": "a + b"},
    write="exp(x - s1) / s2",
    name="my_softmax",
)

x = mx.random.normal((8, 1024))
y = my_softmax(x)
mx.eval(y)

# Verify
ref = mx.softmax(x, axis=-1)
mx.eval(ref)
assert mx.allclose(y, ref, atol=1e-5)
```

## Layer Norm (Map-Reduce Pattern)

Two-pass: compute mean, then variance, then normalize.

```python
from zmlx.api import map_reduce

my_layernorm = map_reduce(
    pass1={"init": "0.0f", "update": "acc1 + x", "reduce": "a + b"},
    pass2={"init": "0.0f", "update": "acc2 + (x - s1/D) * (x - s1/D)", "reduce": "a + b"},
    write="(x - s1/D) * rsqrt(s2/D + 1e-5f)",
    name="my_layernorm",
    threadgroup=256,
)
```

Note: `D` is automatically available as a constant in the generated kernel (the last-dimension size). However, in the current codegen, `s1` and `s2` are the raw reduced values (not divided by D). You need to divide in the expressions as shown above.

## Binary Op with Gradient

A custom binary elementwise op with VJP for both inputs.

```python
from zmlx import autograd
import mlx.core as mx

# Safe multiply: a * b with gradients for both
safe_mul = autograd.binary_from_expr(
    name="safe_mul",
    fwd_expr="a * b",
    vjp_lhs_expr="g * b",
    vjp_rhs_expr="g * a",
    compute_dtype=mx.float32,
)

a = mx.random.normal((1024,))
b = mx.random.normal((1024,))
y = safe_mul(a, b)
```

## N-ary Op with Raw Metal Source

For full control, use `autograd.nary_from_expr()` with raw Metal forward and backward kernels.

```python
from zmlx import autograd
import mlx.core as mx

# Fused: a * sigmoid(b) + c
fused_op = autograd.nary_from_expr(
    name="fused_gate",
    fwd_source="""
        uint elem = thread_position_in_grid.x;
        T a_val = a[elem];
        T b_val = b[elem];
        T c_val = c[elem];
        T sig = T(1) / (T(1) + metal::exp(-b_val));
        out[elem] = a_val * sig + c_val;
    """,
    bwd_source="""
        uint elem = thread_position_in_grid.x;
        T a_val = a[elem];
        T b_val = b[elem];
        T g = cotan[elem];
        T sig = T(1) / (T(1) + metal::exp(-b_val));
        da[elem] = g * sig;
        db[elem] = g * a_val * sig * (T(1) - sig);
        dc[elem] = g;
    """,
    input_names=["a", "b", "c"],
    output_names=["out"],
    compute_dtype=mx.float32,
)
```

## Testing Your Kernel

Use `zmlx.testing` to verify against a reference:

```python
import zmlx.testing
from zmlx.api import elementwise
import mlx.core as mx

_mish_kernel = elementwise("x * tanh(log(1 + exp(x)))", name="test_mish")

def my_mish(x):
    return _mish_kernel(x)

def ref_mish(x):
    return x * mx.tanh(mx.log(1 + mx.exp(x)))

# Value check
zmlx.testing.assert_matches(my_mish, ref_mish, shapes=[(128,), (1024, 512)])

# Gradient check (both must be differentiable)
# zmlx.testing.assert_gradient_matches(my_mish_grad, ref_mish, shapes=[(64, 128)])
```

## Benchmarking

Compare implementations side-by-side:

```python
import zmlx.bench
import mlx.core as mx

zmlx.bench.compare(
    {"Custom": my_op, "MLX Built-in": mx.nn.silu},
    shapes=[(1024, 4096), (4096, 4096)],
    dtypes=[mx.float32, mx.float16],
)
```

Output is a formatted table:

```
               Shape       Dtype        Custom    MLX Built-in     Speedup
------------------------------------------------------------------------
      (1024, 4096)     float32       45.2us        42.1us        0.93x
      (1024, 4096)     float16       38.7us        35.3us        0.91x
      ...
```

## Profiling

Detailed timing and memory statistics:

```python
import zmlx.profile
import mlx.core as mx

x = mx.random.normal((4096, 4096))

# Detailed timing
timing = zmlx.profile.time_kernel(my_op, x, warmup=10, iters=50)
print(f"Median: {timing['median_us']:.1f}us, Std: {timing['std_us']:.1f}us")

# Memory estimate
mem = zmlx.profile.memory_usage(my_op, x)
print(f"Input: {mem['input_bytes']/1e6:.1f}MB, Output: {mem['output_bytes']/1e6:.1f}MB")

# Global kernel statistics
for s in zmlx.profile.kernel_stats():
    print(f"  {s['name']}: {s['run_count']} runs, "
          f"{s['compile_time_ms']:.1f}ms compile")
```
