# ZMLX Quick Start

Go from a math formula to a tested, benchmarked, differentiable Metal kernel in 5 minutes.

## Prerequisites

```bash
pip install zmlx
```

Requires macOS on Apple Silicon (M-series), Python >= 3.10, mlx >= 0.30.0.

## Step 1: Define a Custom Op

### Option A: Using Expression Strings
Let's implement Mish activation: `f(x) = x * tanh(softplus(x))`.

```python
from zmlx.api import elementwise
import mlx.core as mx

# Forward-only kernel (no gradient support)
mish = elementwise("x * tanh(log(1 + exp(x)))", name="mish")

x = mx.random.normal((1024,))
y = mish(x)
mx.eval(y)
print(f"Input shape: {x.shape}, Output shape: {y.shape}")
```

### Option B: Using JIT Decorator
For more complex logic, use the `@zmlx.jit` decorator to compile Python scalar ops.

```python
import zmlx

@zmlx.jit
def mish_jit(x):
    return x * mx.tanh(mx.log(1 + mx.exp(x)))

y = mish_jit(mx.random.normal((1024,)))
mx.eval(y)
```

ZMLX handles Metal source generation, compilation, caching, and launch configuration.

## Step 2: Add Gradient Support

To use the kernel in training, supply a VJP (vector-Jacobian product) expression:

```python
mish_trainable = elementwise(
    "x * tanh(log(1 + exp(x)))",
    name="mish_grad",
    grad_expr="g * (tanh(log(1+exp(x))) + x * (1 - tanh(log(1+exp(x)))*tanh(log(1+exp(x)))) * (1/(1+exp(-x))))",
    use_output=False,
)

# Now it works with mx.grad
loss_fn = lambda z: mish_trainable(z).sum()
gx = mx.grad(loss_fn)(mx.random.normal((1024,)))
mx.eval(gx)
print(f"Gradient shape: {gx.shape}")
```

## Step 3: Verify Correctness

Use `zmlx.testing` to check your kernel against a reference:

```python
import zmlx.testing

def ref_mish(x):
    return x * mx.tanh(mx.log(1 + mx.exp(x)))

zmlx.testing.assert_matches(
    mish, ref_mish,
    shapes=[(128,), (1024,), (8, 512)],
)
print("All correctness checks passed!")
```

## Step 4: Benchmark

Compare against the reference implementation:

```python
import zmlx.bench

zmlx.bench.compare(
    {"ZMLX Mish": mish, "MLX Ref": ref_mish},
    shapes=[(1024,), (1024, 4096), (4096, 4096)],
)
```

This prints a formatted table with latency and speedup for each shape and dtype.

## Step 5: Inspect the Generated Metal

See exactly what Metal code ZMLX generated:

```python
import zmlx.profile

stats = zmlx.profile.kernel_stats()
for s in stats:
    print(f"  {s['name']}: compiled in {s['compile_time_ms']:.1f}ms, "
          f"ran {s['run_count']} times")
```

## What's Next?

- **JIT Compiler**: `@zmlx.jit` for compiling Python functions to fused kernels
- **Fused AdamW**: `zmlx.optimizers.AdamW` — single-kernel optimizer step
- **Paged Attention**: `zmlx.kernels.attention.paged_attention` for serving
- **Custom reductions**: `zmlx.api.reduce()` for rowwise ops
- **Map-reduce patterns**: `zmlx.api.map_reduce()` for complex fusions

See [`COOKBOOK.md`](COOKBOOK.md) for recipes covering all these patterns.
