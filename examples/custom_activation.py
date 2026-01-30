"""Example: Custom differentiable activation with ZMLX.

Demonstrates:
- Creating an elementwise kernel from a C expression
- Adding gradient support via VJP
- Verifying against a reference implementation
- Benchmarking
"""

import mlx.core as mx

import zmlx.bench
import zmlx.testing
from zmlx.api import elementwise

# --- 1. Define the kernel ---
# Mish activation: f(x) = x * tanh(softplus(x))
mish = elementwise(
    "x * tanh(log(1 + exp(x)))",
    name="mish",
)

# --- 2. Run it ---
x = mx.random.normal((8, 1024))
y = mish(x)
mx.eval(y)
print(f"Mish output shape: {y.shape}, dtype: {y.dtype}")

# --- 3. Reference implementation ---
def ref_mish(x: mx.array) -> mx.array:
    return x * mx.tanh(mx.log(1 + mx.exp(x)))

# --- 4. Verify correctness ---
zmlx.testing.assert_matches(
    mish, ref_mish,
    shapes=[(128,), (1024,), (8, 512), (32, 2048)],
    dtypes=[mx.float32],
)
print("Correctness: PASSED")

# --- 5. Benchmark ---
print("\nBenchmark:")
zmlx.bench.compare(
    {"ZMLX Mish": mish, "MLX Ref": ref_mish},
    shapes=[(1024,), (1024, 4096), (4096, 4096)],
    dtypes=[mx.float32],
)
