# ZMLX Kernel Catalog

This document lists the custom Metal kernels available in the `zmlx.kernels` package.

> **Note on performance**: These catalog kernels serve two purposes:
>
> 1. **Genuinely useful ops** that have no direct MLX equivalent or provide real
>    fusion benefits (e.g. `softmax_cross_entropy`, `swiglu`, `pack_bits`,
>    `top2_gating_softmax`, `moe_dispatch`, `moe_combine`, `cumsum_lastdim`).
>
> 2. **Reference implementations** that demonstrate ZMLX codegen patterns
>    (parallel reduction, map-reduce, elementwise with VJP). These are correct
>    and serve as starting points for custom ops.
>
> **At the op level**, MLX built-ins like `mx.fast.rms_norm` and `mx.softmax`
> are faster than standalone ZMLX equivalents due to lower dispatch overhead.
> **At the model level**, fusing operations (e.g. `rmsnorm_residual` fuses
> residual-add + RMSNorm into one pass) saves memory bandwidth and delivers
> +33% decode on large dense models and +36% on MoE models.
> Use `zmlx.patch.patch(model)` to apply these fusions automatically.

## Activations (`zmlx.kernels.activations`)

| Kernel | Description | Gradient Support |
|:--- |:--- |:--- |
| `exp` | Elementwise exponential | Yes (`exp_grad`) |
| `log` | Elementwise natural logarithm | No |
| `tanh` | Elementwise hyperbolic tangent | Yes (`tanh_grad`) |
| `sigmoid` | Elementwise sigmoid | Yes (`sigmoid_grad`) |
| `relu` | Elementwise ReLU | Yes (`relu_grad`) |
| `silu` | Elementwise SiLU (swish) | Yes (`silu_grad`) |
| `gelu_tanh` | Elementwise GeLU (tanh approximation) | Yes (`gelu_tanh_grad`) |
| `softplus` | Elementwise softplus | Yes (`softplus_grad`) |
| `mish` | Elementwise mish | Yes (`mish_grad`) |
| `elu` | Elementwise exponential linear unit | Yes (`elu_grad`) |

## Softmax (`zmlx.kernels.softmax`)

| Kernel | Description |
|:--- |:--- |
| `softmax_lastdim(x)` | Softmax over last dimension (differentiable) |
| `log_softmax_lastdim(x)` | Stable Log-Softmax over last dimension |
| `softmax_grad(y, cotan)` | Backward helper for softmax (internal use) |

## Norms (`zmlx.kernels.norms`)

| Kernel | Description |
|:--- |:--- |
| `rmsnorm(x, w)` | RMSNorm over last dimension (differentiable) |
| `rmsnorm_grad(x, w, cotan)` | Backward helper for RMSNorm (internal use) |
| `rms_norm_no_weight(x)` | RMSNorm without weights |
| `layernorm(x, g, b)` | LayerNorm over last dimension |
| `layer_norm_no_weight(x)` | LayerNorm without weights |
| `layer_norm_dropout(x, g, b, p, seed)` | Fused LayerNorm + Dropout |

## Transformer Essentials (`zmlx.kernels.transformer`)

| Kernel | Description |
|:--- |:--- |
| `swiglu(x)` | Fused SiLU(x1) * x2 where x is [..., 2D] |
| `geglu(x)` | Fused GeLU(x1) * x2 where x is [..., 2D] |
| `rmsnorm_residual(x, res, w)` | Fused RMSNorm(x + residual) and returns (normed, x+res) |
| `layernorm_residual(x, res, g, b)` | Fused LayerNorm(x + residual) and returns (normed, x+res) |
| `fused_add_rmsnorm(x1, x2, w)` | Fused RMSNorm(x1 + x2) returning only normed output |
| `fused_add_layernorm(x1, x2, g, b)` | Fused LayerNorm(x1 + x2) returning only normed output |
| `rms_norm_dropout(x, w, p, seed)` | Fused RMSNorm + Dropout |
| `dropout(x, p, seed)` | Fused dropout with Metal-side LCG RNG |
| `bias_swiglu(x, bias)` | Fused (x + bias) -> SwiGLU |
| `bias_geglu(x, bias)` | Fused (x + bias) -> GeGLU |

## VLSP (`zmlx.kernels.vlsp`)

| Kernel | Description |
|:--- |:--- |
| `fused_recurrent_step(h, w_norm, gate, alpha)` | Fused RMSNorm + SiLU gating + residual update |
| `depth_gate_sigmoid(x, k_max)` | Differentiable depth prediction with STE backward |
| `grpo_advantage_norm(rewards)` | Fused per-group advantage normalization (GRPO) |
| `silu_mul_residual(gate, up, residual)` | Fused silu(gate) * up + residual (custom VJP) |

## Attention (`zmlx.kernels.attention`)

| Kernel | Description |
|:--- |:--- |
| `logsumexp_lastdim(x)` | Numerically stable LogSumExp |
| `masked_softmax(x, mask)` | Softmax with boolean masking |
| `scale_mask_softmax(x, mask, scale)` | Fused (x * scale) + mask gating -> Softmax |
| `paged_attention(...)` | vLLM-style paged attention for high-throughput serving |
| `attention_tile_proto(q, k)` | 16x16 attention tile prototype (experimental) |

## RoPE (`zmlx.kernels.rope`)

| Kernel | Description |
|:--- |:--- |
| `apply_rope(x, cos, sin)` | Rotary Positional Embedding (half-rot) |
| `apply_rope_interleaved(x, cos, sin)` | Rotary Positional Embedding (interleaved) |
| `apply_gqa_rope(x, cos, sin, n_kv_heads)` | RoPE for Grouped Query Attention |
| `rope_and_cache_update(...)` | Fused RoPE application + KV cache update (contiguous) |
| `paged_rope_and_cache_update(...)` | Fused RoPE application + KV cache update (paged) |

## Reductions (`zmlx.kernels.reductions`)

All reductions operate over the **last dimension**.

| Kernel | Description |
|:--- |:--- |
| `sum_lastdim(x)` | Sum reduction (Kahan support) |
| `mean_lastdim(x)` | Arithmetic mean (Kahan support) |
| `max_lastdim(x)` | Maximum value |
| `var_lastdim(x)` | Variance |
| `std_lastdim(x)` | Standard deviation |
| `argmax_lastdim(x)` | Index of the maximum value (returns `uint32`) |
| `topk_lastdim(x, k)` | Top-K values and indices (small K optimization) |

## Fused Elementwise (`zmlx.kernels.fused`)

| Kernel | Description |
|:--- |:--- |
| `add(a, b)` | Elementwise add |
| `mul(a, b)` | Elementwise multiply |
| `add_bias(x, bias)` | Add 1D bias over last dimension |
| `bias_gelu_tanh(x, bias)` | Fused bias-add + GeLU (tanh approx) |
| `bias_silu(x, bias)` | Fused bias-add + SiLU |
| `silu_mul_grad(a, b)` | Fused SiLU(a) * b with custom VJP |

## Linear (`zmlx.kernels.linear`) — Reference Implementations

> These use naive dot-product matmul (one thread per output element) and are
> **not competitive** with MPS-accelerated `mx.matmul`. They demonstrate how to
> fuse post-linear operations into a single Metal kernel.

| Kernel | Description |
|:--- |:--- |
| `fused_linear_bias_silu(x, w, b)` | Linear + Bias + SiLU (reference) |
| `fused_linear_bias_gelu(x, w, b)` | Linear + Bias + GeLU (reference) |
| `fused_linear_rmsnorm(x, w, g)` | Linear + RMSNorm (reference) |

## Loss Functions (`zmlx.kernels.loss`)

| Kernel | Description |
|:--- |:--- |
| `softmax_cross_entropy(logits, targets)` | Fused Softmax + Cross Entropy Loss |

## Quantization (`zmlx.kernels.quant`)

| Kernel | Description |
|:--- |:--- |
| `dequantize_int8(x, scale)` | y = x * scale (int8 to float) |
| `dequantize_silu_int8(x, scale)` | Fused dequantize + SiLU |
| `dequantize_int4(x, scale)` | y = x * scale (int4 packed in uint8) |

## Bit Ops (`zmlx.kernels.bits`)

| Kernel | Description |
|:--- |:--- |
| `pack_bits(x)` | Pack bool/uint8 array into bits (uint8) |
| `unpack_bits(x)` | Unpack bits into uint8 array |

## Mixture of Experts (`zmlx.kernels.moe`)

| Kernel | Description |
|:--- |:--- |
| `top2_gating_softmax(x)` | Select top 2 experts and return weights + indices |
| `moe_dispatch(x, indices)` | Fused token-to-expert dispatch |
| `moe_combine(experts, weights)` | Fused expert-output combination and weighting |

## Image (`zmlx.kernels.image`)

| Kernel | Description |
|:--- |:--- |
| `resize_bilinear(x, output_shape)` | Bilinear resize for NHWC images |
| `depthwise_conv_3x3(x, w)` | 3x3 depthwise convolution (NHWC) |

## Indexing (`zmlx.kernels.indexing`)

| Kernel | Description |
|:--- |:--- |
| `fused_gather_add(src, idx, other)` | Gather rows from src and add elementwise other |
| `fused_scatter_add(idx, updates, shape)` | Scatter updates into a zero-init array with atomics |

## Optimizers (`zmlx.kernels.optimizers`)

| Kernel | Description |
|:--- |:--- |
| `adamw_step(...)` | Fused AdamW update (p, m, v update in one pass) |

## Scan (`zmlx.kernels.scan`)

| Kernel | Description |
|:--- |:--- |
| `cumsum_lastdim(x)` | Inclusive prefix sum over the last dimension (differentiable) |
| `cumsum_grad(cotan)` | Backward helper for cumsum (internal use) |
