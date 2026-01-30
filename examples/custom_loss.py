"""Example: Custom loss function with ZMLX.

Demonstrates:
- Creating a rowwise reduction kernel
- Using it as a loss function
"""

import mlx.core as mx

from zmlx.api import reduce

# --- 1. Entropy loss: -sum(p * log(p + eps)) ---
entropy = reduce(
    init="0.0f",
    update="acc + (-v * log(v + 1e-8f))",
    name="entropy",
)

# Create probability distributions
logits = mx.random.normal((4, 1024))
probs = mx.softmax(logits, axis=-1)

h = entropy(probs)
mx.eval(h)
print(f"Entropy per row: {h}")

# --- 2. Reference implementation ---
def ref_entropy(p: mx.array) -> mx.array:
    return (-p * mx.log(p + 1e-8)).sum(axis=-1)

ref = ref_entropy(probs)
mx.eval(ref)
print(f"Reference:        {ref}")

# Check
diff = mx.abs(h - ref)
mx.eval(diff)
print(f"Max diff: {mx.max(diff).item():.2e}")

# --- 3. Row-max reduction ---
row_max = reduce(
    init="-INFINITY",
    update="max(acc, v)",
    name="row_max",
)

x = mx.random.normal((4, 512))
maxvals = row_max(x)
ref_max = x.max(axis=-1)
mx.eval(maxvals, ref_max)
print(f"\nRow max:     {maxvals}")
print(f"Reference:   {ref_max}")
print(f"Match: {mx.allclose(maxvals, ref_max, atol=1e-5).item()}")
