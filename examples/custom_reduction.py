"""Example: Custom map-reduce kernel with ZMLX.

Demonstrates:
- Creating a two-pass map-reduce kernel (softmax pattern)
- Verifying against MLX built-in
- Benchmarking the comparison
"""

import mlx.core as mx

import zmlx.bench
import zmlx.testing
from zmlx.api import map_reduce

# --- 1. Custom softmax via map-reduce ---
# Pass 1: find row max
# Pass 2: sum of exp(x - max)
# Write: exp(x - max) / sum
my_softmax = map_reduce(
    pass1={"init": "-INFINITY", "update": "max(acc1, x)", "reduce": "max(a, b)"},
    pass2={"init": "0.0f", "update": "acc2 + exp(x - s1)", "reduce": "a + b"},
    write="exp(x - s1) / s2",
    name="my_softmax",
)

x = mx.random.normal((8, 1024))
y = my_softmax(x)
mx.eval(y)
print(f"Softmax output shape: {y.shape}")
print(f"Row sums (should be ~1.0): {y.sum(axis=-1)}")

# --- 2. Verify against MLX ---
def ref_softmax(x: mx.array) -> mx.array:
    return mx.softmax(x, axis=-1)

zmlx.testing.assert_matches(
    my_softmax, ref_softmax,
    shapes=[(4, 256), (8, 1024), (32, 4096)],
    dtypes=[mx.float32],
)
print("\nCorrectness: PASSED")

# --- 3. Benchmark ---
print("\nBenchmark:")
zmlx.bench.compare(
    {"ZMLX Softmax": my_softmax, "MLX Softmax": ref_softmax},
    shapes=[(8, 1024), (32, 4096), (128, 4096)],
    dtypes=[mx.float32],
)
