# ZMLX

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform: macOS Apple Silicon](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg)](https://github.com/ml-explore/mlx)

Numba-style helpers for authoring, autotuning, and differentiating custom Metal kernels on Apple silicon.

```bash
pip install zmlx
```

```python
from zmlx import autograd, msl
import mlx.core as mx

silu = autograd.unary_from_expr(
    name="my_silu", fwd_expr="x * kk_sigmoid(x)",
    vjp_expr="g * (s + x * s * ((T)1 - s))",
    compute_dtype=mx.float32, use_output=False,
    vjp_prelude="T s = kk_sigmoid(x);",
    header=msl.DEFAULT_HEADER,
)
y = silu(mx.random.normal((1024,)))  # runs on GPU, supports mx.grad
```

---

## Why ZMLX?

- **Define-once caching** — kernels compile once and are reused across calls (keyed on source hash + config).
- **One-line ops** — create elementwise kernels from a C expression: `elementwise.unary(name=..., expr=...)`.
- **Differentiable kernels** — attach custom VJP backward passes (themselves Metal kernels) via `mx.custom_function`.
- **Autotuning** — search threadgroup sizes automatically and cache the winners.
- **70+ ready-to-use catalog kernels** — activations, softmax, norms, RoPE, transformer fused ops, reductions, quantization, and more.

---

## Install

**Requirements**: macOS (Apple Silicon), Python >= 3.10, mlx >= 0.30.0

```bash
pip install zmlx
```

From source (development):

```bash
git clone https://github.com/Hmbown/ZMLX.git
cd ZMLX
pip install -e ".[dev]"
```

---

## Quick Examples

### 1. Elementwise kernel from a C expression

```python
from zmlx import elementwise, msl
import mlx.core as mx

exp_fast = elementwise.unary(
    name="kk_exp",
    expr="metal::exp(x)",
    compute_dtype=mx.float32,
    header=msl.DEFAULT_HEADER,
)

x = mx.random.normal((1024,)).astype(mx.float16)
y = exp_fast(x)
```

### 2. Differentiable kernel with custom VJP

```python
from zmlx import autograd, msl
import mlx.core as mx

exp_trainable = autograd.unary_from_expr(
    name="kk_exp_vjp",
    fwd_expr="metal::exp(x)",
    vjp_expr="g * y",
    compute_dtype=mx.float32,
    use_output=True,
    header=msl.DEFAULT_HEADER,
)

def loss(z):
    return exp_trainable(z).sum()

x = mx.random.normal((1024,))
gx = mx.grad(loss)(x)
mx.eval(gx)
```

### 3. Catalog kernel (ready-to-use)

```python
from zmlx.kernels import softmax, norms, transformer
import mlx.core as mx

x = mx.random.normal((8, 1024)).astype(mx.float16)
w = mx.ones((1024,), dtype=mx.float16)

y = softmax.softmax_lastdim(x)           # rowwise softmax
z = norms.rmsnorm(x, w)                   # differentiable RMSNorm
s = transformer.swiglu(mx.random.normal((8, 2048)).astype(mx.float16))  # fused SwiGLU
```

### 4. High-level UX API (load / LoRA / train / generate)

Requires `mlx-lm`: `pip install 'zmlx[train]'`

```python
import zmlx

# Load a model with ZMLX fused-kernel patching applied automatically
model, tokenizer = zmlx.load("mlx-community/Llama-3.2-1B-4bit", patch=True)

# Apply LoRA adapters
zmlx.lora(model, r=16, alpha=16.0)

# Fine-tune
zmlx.train(model, tokenizer, "path/to/dataset", iters=500)

# Generate
print(zmlx.generate(model, tokenizer, "Once upon a time"))
```

---

## Kernel Catalog

18 modules, 70+ kernels (including gradient helpers) organized by domain. Full reference: [`docs/KERNELS.md`](docs/KERNELS.md).

| Module | Count | Highlights |
|:---|:---:|:---|
| `activations` | 19 | exp, sigmoid, relu, silu, gelu_tanh, softplus + grad variants |
| `transformer` | 10 | swiglu, geglu, rmsnorm_residual, layernorm_residual, dropout |
| `vlsp` | 4 | fused_recurrent_step, depth_gate_sigmoid, grpo_advantage_norm |
| `softmax` | 3 | softmax_lastdim, log_softmax_lastdim, softmax_grad |
| `norms` | 6 | rmsnorm, layernorm, rmsnorm_grad, layer_norm_dropout |
| `attention` | 4 | masked_softmax, scale_mask_softmax, logsumexp_lastdim |
| `rope` | 3 | apply_rope, apply_rope_interleaved, apply_gqa_rope |
| `reductions` | 7 | sum, mean, max, var, std, argmax, topk (all lastdim) |
| `fused` | 6 | add, mul, bias_gelu_tanh, bias_silu, silu_mul_grad, add_bias |
| `linear` | 4 | fused_linear_bias_silu, fused_linear_bias_gelu, fused_linear_rmsnorm |
| `loss` | 1 | softmax_cross_entropy |
| `quant` | 3 | dequantize_int8, dequantize_silu_int8, dequantize_int4 |
| `bits` | 2 | pack_bits, unpack_bits |
| `moe` | 1 | top2_gating_softmax |
| `image` | 2 | resize_bilinear, depthwise_conv_3x3 |
| `indexing` | 2 | fused_gather_add, fused_scatter_add |
| `scan` | 2 | cumsum_lastdim, cumsum_grad |

---

## Architecture

Three-layer design. Full details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

1. **Metal kernel infrastructure** — `MetalKernel` wrapper, in-process cache, stats tracking
2. **Code generation & helpers** — MSL templates, elementwise/autograd/rowwise APIs, autotuning
3. **Kernel catalog** — 17 domain modules built on layers 1 and 2

---

## Benchmarks

Run `python benchmarks/microbench.py` to reproduce. Headline numbers on M4 (vs MLX reference ops):

- **Dropout**: ~7x faster (fused Metal-side LCG RNG)
- **SwiGLU**: ~2-3x faster (fused silu + gate multiply)

Results vary by shape, dtype, and chip. See [`benchmarks/`](benchmarks/) for the full harness.

---

## Roadmap

- Flash Attention tiles (shared memory, 16x16 / 32x32)
- Expanded quantization (int4 matmul, mixed-precision patterns)
- Zig frontend via C++ shim (MLX-C once available)
- JVP support for all catalog kernels
- Community-contributed kernels

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, testing, and conventions.

---

## License

MIT. See [`LICENSE`](LICENSE).
