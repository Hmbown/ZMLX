# HCSA Integration Guide for ZMLX

Sparse attention via Hamiltonian cycles applied to ZMLX-patched models on Apple Silicon.

**HCSA repo:** `/Volumes/VIXinSSD/wayfinder` (package `hcsa`, [github.com/Hmbown/hcsa](https://github.com/Hmbown/hcsa))
**ZMLX repo:** `/Volumes/VIXinSSD/ZMLX` (package `zmlx`, [github.com/Hmbown/ZMLX](https://github.com/Hmbown/ZMLX))

---

## 1. What HCSA Does (and Doesn't Do)

HCSA replaces **dense causal self-attention** O(T^2) with **sparse attention** O(T * W) by
building a Hamiltonian cycle over token positions and attending only within a local window
in cycle-permuted space, plus optional landmark and rewire edges.

| Dimension | HCSA | ZMLX |
|---|---|---|
| **Target bottleneck** | Attention (QKV matmul, softmax) | MoE FFN (gating, expert dispatch, activation) |
| **Phase** | Prefill (long context) | Decode (single token) |
| **Mechanism** | Sparse attention graphs | Fused Metal kernels |
| **Modules touched** | `self_attn` | MoE block, norms, activations |
| **Numerical equivalence** | No (sparse approximation) | Yes (token-identical) |

**They are complementary, not competing.** HCSA swaps attention modules; ZMLX patches MoE/FFN
modules. Both can be applied to the same model simultaneously since they target non-overlapping
submodules.

### What HCSA gives you

- **Memory**: 26% reduction at T=65536 (measured on GLM-4.7-Flash-4bit, M4 Max)
- **Latency**: O(T * W) instead of O(T^2) for prefill. Decode is unchanged (Q_len=1 is already O(T))
- **Quality**: Hamiltonian cycle guarantees every token is reachable within O(log T) hops with
  high probability (random strategy). Spectral gap verification available.

### Known limitation: chunked prefill dense fallback

`glm_mlx.py:259` — when Q_len != K_len (all prefill chunks after the first), the current code
falls back to O(T^2) dense attention. The fix (scatter Q into permuted space, run local-window
attention at active positions, gather output) is partially implemented via
`wayfinder_permute_window_attention_active_batched()`. This is the root cause of the 6.9x
latency regression at T=65536 with chunk=4096. Memory reduction is real and unaffected.

---

## 2. Shared Models: Where Both Apply Today

| Model | ZMLX pattern | HCSA integration | Combined opportunity |
|---|---|---|---|
| **GLM-4.7-Flash-4bit** | `moe_mlp` (+6.6-8.5% decode) | `GLMWayfinderAttention` (26% mem at 65k) | Decode speedup + prefill mem savings |
| **Qwen3-30B-A3B-4bit** | `moe_mlp` (+7.9-8.1% decode) | `QwenWayfinderAttention` | Decode speedup + prefill mem savings |
| **LFM2-8B-A1B-4bit** | `moe_mlp` (+12.8% decode) | None yet | ZMLX only for now |
| **DeepSeek-V3.2 / Kimi-K2.5** | `deepseek_router` (experimental) | None yet | ZMLX only for now |
| **GPT-OSS-20B** | `swiglu_mlp` (+1% decode) | None yet | ZMLX only for now |

**Immediate value: GLM-4.7-Flash and Qwen3-30B** — apply both `zmlx.patch()` and HCSA swap
on the same model instance.

---

## 3. Architecture: How They Compose

```
Model loaded from mlx-community hub
    |
    v
[1] zmlx.patch(model, profile="glm")        # patches MoE blocks (FFN)
    |  - replaces moe_block.__call__ with fused gating + combine
    |  - touches: switch_mlp, gate/router, shared_experts
    |  - does NOT touch: self_attn
    |
    v
[2] swap_glm_attention_with_wayfinder(       # patches attention layers
        model,
        cfg=GLMWayfinderConfig(window=64, strategy="random"),
        layer_indices=None,                  # all 47 layers
    )
    |  - replaces layer.self_attn with GLMWayfinderAttention
    |  - does NOT touch: MoE block, norms, activations
    |
    v
Model ready: MoE decode fused by ZMLX, attention sparsified by HCSA
```

**Order doesn't matter** — the swaps target different child attributes on each transformer layer.
`zmlx.patch()` matches on MoE block class names; HCSA swap matches on `self_attn` attribute.

### Module-level non-interference proof

A GLM transformer layer (`TransformerBlock`) has this structure:
```
TransformerBlock
  ├── self_attn: Attention          ← HCSA replaces this
  ├── mlp: SparseMoeBlock           ← ZMLX patches this
  ├── input_layernorm: RMSNorm
  └── post_attention_layernorm: RMSNorm
```

ZMLX's `_MoEMLPPattern.matches()` checks for `gate`/`router` + `experts`/`switch_mlp` attributes.
HCSA's `swap_*_attention_with_wayfinder()` replaces `layer.self_attn` by attribute name. No overlap.

---

## 4. HCSA Core Concepts

### 4.1 The Sparse Attention Graph

Each token `i` attends to a small set of neighbors instead of all previous tokens:

1. **Cycle neighbors** — 2 adjacent nodes in a Hamiltonian cycle (undirected; causality enforced
   by masking `j > i`)
2. **Local causal window** — `{max(0, i-w), ..., i-1}` (always included)
3. **Landmarks** (optional) — every k-th token: `{j | j % stride == 0 and j < i}`
4. **Self** — always included

Total per-token degree D is typically 64-256, vs T for dense attention.

### 4.2 The Graph ABI

**File:** `hcsa/graph/abi.py`

The language-agnostic bridge between graph construction (CPU/Python) and attention kernels (MLX).

```python
@dataclass(frozen=True)
class WayfinderGraphABI:
    neigh_idx: np.ndarray   # [T, D] or [H, T, D], int32, -1 = padding
    edge_type: np.ndarray   # same shape, uint8
    meta: Dict[str, Any]    # cycle_perms, all_cycle_perms, seq_len, etc.
```

**Edge types** (`EdgeType(IntEnum)`):
| Value | Name | Meaning |
|---|---|---|
| 0 | `PAD` | Padding (invalid neighbor) |
| 1 | `CYCLE` | Hamiltonian cycle edge |
| 2 | `WINDOW` | Local causal window |
| 3 | `LANDMARK` | Periodic landmark |
| 4 | `REWIRE` | Query-driven rewiring |

**Edge priority** (for conflict resolution when two edge types compete for same slot):
`WINDOW (0) < LANDMARK (1) < REWIRE (2) < CYCLE (3)` — cycle edges win.

**Key functions:**
- `build_graph_abi_from_adjacency(*, T, cycle_adj, window, landmark_stride, ...)` — core builder
- `stack_head_abis(head_abis)` — merges H single-head `[T, D]` ABIs into `[H, T, D]`
- `validate_graph_abi(abi, *, expect_heads, expect_tokens, enforce_hamiltonian)` — shape/invariant checks
- `graph_metrics(abi, *, bfs_hops=4)` — degree stats, shortcut rate, BFS reachability proxy

**MLX conversion** (`hcsa/mlx/graph_abi.py`):
```python
@dataclass(frozen=True)
class MLXGraphABI:
    neigh_idx: mx.array   # [H, T, D] int32
    edge_type: mx.array   # [H, T, D] uint8
    meta: Dict[str, Any]
```
- `to_mlx_graph_abi(abi)` — numpy → mx.array
- `safe_neighbor_idx(neigh_idx, seq_len)` — clamps `-1` to `0` for safe gather
- `causal_neighbor_mask(neigh_idx, seq_len)` — `True` for valid causal neighbors

### 4.3 Cycle Strategies

**File:** `hcsa/graph_strategies.py`

**Protocol:**
```python
class GraphStrategy(Protocol):
    def build_adjacency(self, T, r=None, head_idx=0) -> List[List[int]]: ...
    def build(self, T, r=None, head_idx=0, *, window, landmark_stride, include_self) -> WayfinderGraphABI: ...
```

| Strategy | Complexity | Input-dependent? | Best for |
|---|---|---|---|
| `random` | O(T) | No (static) | Inference, cacheable |
| `greedy` | O(T^2) | Yes (needs routing embeddings) | Training, quality-critical |
| `online_insertion` | O(T) | Yes (incremental) | Streaming |
| `regular_partition` | O(T) | No (static) | Balanced degree |

**For ZMLX integration, `random` is the right choice** — it's input-independent (cacheable across
calls), O(T) construction, and the spectral gap guarantees O(log T) reachability with high
probability.

**Registry:**
```python
STRATEGY_REGISTRY = {
    "random": RandomCycleStrategy,
    "greedy": GreedyCycleStrategy,
    "online_insertion": OnlineInsertionStrategy,
    "regular_partition": RegularPartitionStrategy,
}
build_strategy(name, **kwargs) -> GraphStrategy
```

### 4.4 Topology Runtime

**File:** `hcsa/topology/core.py`

Orchestrates per-head strategy instances and graph caching.

```python
class Topology:
    def __init__(self, *, n_heads, strategy="random", num_cycles=1,
                 edge_disjoint=True, seed=0, window=64, landmark_stride=64, ...): ...

    def construct_abi(self, *, T, routing_by_head=None, include_self=True) -> WayfinderGraphABI:
        # Calls strategy.build() per head, then stack_head_abis()
        ...

    @property
    def cache_mode(self) -> str:
        # "static" for random/regular_partition, "dynamic" for greedy/online
        ...
```

Per-head seed diversification: `seed + 7919 * head_idx` ensures different cycles per head.

---

## 5. MLX Attention Kernels

**File:** `hcsa/mlx/attention.py`

### 5.1 Dense Causal (fallback)

```python
def dense_causal_attention(
    q: mx.array,    # [B, H, T, dh]
    k: mx.array,    # [B, H, T, dh]
    v: mx.array,    # [B, H, T, dh]
    *,
    return_weights: bool = False,
) -> Tuple[mx.array, Optional[mx.array]]:
```
Uses `mx.fast.scaled_dot_product_attention(mask="causal")` when available.

### 5.2 Sparse Gather (reference path)

```python
def sparse_gather_attention(
    q: mx.array,                          # [B, H, T, dh]
    k: mx.array,                          # [B, H, T, dh]
    v: mx.array,                          # [B, H, T, dh]
    graph: MLXGraphABI,
    *,
    return_weights: bool = False,
    precomputed_safe_idx: Optional[mx.array] = None,    # [H, T, D]
    precomputed_causal_mask: Optional[mx.array] = None,  # [H, T, D]
    edge_type_bias: Optional[mx.array] = None,           # [4] learnable
    edge_type_bias_offset: Optional[mx.array] = None,    # [4]
    window_drop_mask: Optional[mx.array] = None,          # [H, T, D] bool
) -> Tuple[mx.array, Optional[mx.array]]:
```
- Gathers K/V at neighbor indices per head
- Dot-product scores + optional edge-type bias
- Causal + validity masked softmax
- Weighted sum over D neighbors

### 5.3 Permute Window (fast path — production)

```python
def wayfinder_permute_window_attention_batched(
    q: mx.array,                  # [B, Hq, T, dh]
    k: mx.array,                  # [B, Hkv, T, dh]
    v: mx.array,                  # [B, Hkv, T, dh]
    *,
    all_perms: mx.array,          # [Hq, T] or [Hq, d, T] (d = num_cycles)
    all_inv_perms: mx.array,      # [Hq, T] or [Hq, d, T]
    window: int = 64,
    edge_type_bias_scalar: Optional[float] = None,
    window_drop_prob: float = 0.0,
    training: bool = False,
    head_chunk_size: Optional[int] = None,
    query_chunk_size: int = 256,
    prepermute_mode: Literal["auto", "off", "kv", "qkv", "on"] = "auto",
    memory_budget_bytes: Optional[int] = None,
    retro_backfill_enabled: bool = False,
    retro_backfill_alpha: float = 0.0,
    retro_backfill_training_only: bool = True,
    retro_backfill_causal_only: bool = True,
    log_progress: bool = False,
) -> Tuple[mx.array, Optional[mx.array]]:
```

**Algorithm:**
1. Permute Q/K/V into cycle order using pre-computed permutation
2. Gather local window `[T, 2*window+1]` around each permuted position
3. Compute attention as local causal masked softmax
4. Inverse permute output back to original token order

**Key features:**
- GQA support: `Hq % Hkv == 0`, maps query heads to KV heads
- Multi-cycle: `all_perms` 3D `[H, d, T]` → averages across d cycles
- Head chunking: `mx.eval(y_h)` between chunks to bound memory
- Query chunking: processes `query_chunk_size` permuted positions per step
- Prepermute planner: cost model selects off/kv/qkv based on memory budget
- Uses `mx.fast.scaled_dot_product_attention` when possible

### 5.4 Active Query (chunked prefill)

```python
def wayfinder_permute_window_attention_active_batched(
    q: mx.array,                  # [B, Hq, Tq, dh]
    k: mx.array,                  # [B, Hkv, Tk, dh]
    v: mx.array,                  # [B, Hkv, Tk, dh]
    *,
    all_perms: mx.array,          # [Hq, Tg] (Tg >= Tk, graph horizon)
    all_inv_perms: mx.array,      # [Hq, Tg]
    query_positions: mx.array,    # [Tq] original token positions
    window: int = 64,
    edge_type_bias_scalar: Optional[float] = None,
    head_chunk_size: Optional[int] = None,
    query_chunk_size: int = 256,
    prepermute_mode: Literal["auto", "off", "kv", "qkv", "on"] = "auto",
    memory_budget_bytes: Optional[int] = None,
    log_progress: bool = False,
) -> Tuple[mx.array, Optional[mx.array]]:
```
For Q_len < K_len (prefill chunks after the first):
1. Maps query positions to cycle rank via inverse permutation
2. Builds window of ranks around each query's cycle position
3. Maps ranks back to original indices
4. Masks: valid (within graph), available (< Tk), causal (k_orig <= q_pos)

### 5.5 Utility: Stable Masked Softmax

```python
def stable_masked_softmax(
    scores_f32: mx.array,
    mask: mx.array,
    axis: int = -1,
    *,
    preserve_dtype: bool = False,
) -> mx.array:
```
Fills masked positions with -1e30 (or -1e4 for float16), subtracts row max, exponentiates,
zeros masked, normalizes.

---

## 6. Existing Model Integrations (Reference Implementations)

### 6.1 Qwen3 Integration

**File:** `hcsa/integrations/qwen_mlx.py`

**Config:**
```python
@dataclass
class QwenWayfinderConfig:
    path: Literal["sparse", "permute"] = "permute"
    strategy: Literal["random", "greedy", "online_insertion", "regular_partition"] = "random"
    window: int = 64
    landmark_stride: Optional[int] = 64
    num_cycles: int = 1
    edge_disjoint: bool = True
    regular_num_clusters: int = 8
    seed: int = 0
    edge_bias: bool = True
    window_drop: float = 0.0
    compiled_graph_dir: Optional[str] = None
    permute_head_chunk_size: int = 8
    query_chunk_size: int = 256
    permute_stream_o_proj: bool = False
    permute_log_chunks: bool = False
    compute_edge_utilization_proxy: bool = True
    compute_graph_metrics: bool = True
    retro_backfill_enabled: bool = False
    retro_backfill_alpha: float = 0.0
    retro_backfill_training_only: bool = True
    retro_backfill_causal_only: bool = True
    verify_spectral_gap: bool = False
    spectral_gap_threshold: float = 4.0
```

**QKV extraction:**
```python
def extract_qkv_from_qwen_attention(attn, x, *, cache=None):
    # Returns (queries [B, Hq, Tq, Dh], keys [B, Hk, Tk, Dh], values [B, Hk, Tk, Dh])
    # Applies q_proj/k_proj/v_proj, q_norm/k_norm, RoPE, optional KV cache update
```

**Wrapper class:** `QwenWayfinderAttention(nn.Module)`
- Copies projections from base: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `q_norm`, `k_norm`, `rope`
- Creates `_QwenGraphRuntime` for graph construction/caching
- Optional `edge_type_bias = mx.zeros((4,))` learnable parameter
- `__call__(x, mask=None, cache=None)`:
  1. Extracts Q/K/V via `extract_qkv_from_qwen_attention`
  2. Falls back to dense when Q_len != K_len (incremental decode)
  3. Gets/builds graph cache (two-level: per-instance + by-key sharing)
  4. Dispatches to sparse or permute path
  5. Applies `o_proj`

**Swap function:**
```python
def swap_qwen_attention_with_wayfinder(
    model: nn.Module,
    *,
    cfg: QwenWayfinderConfig,
    layer_indices: Optional[List[int]] = None,  # None = all layers
) -> List[int]:
    # Iterates model.layers, replaces layer.self_attn
    # Returns indices of replaced layers
```

### 6.2 GLM-4.7-Flash Integration

**File:** `hcsa/integrations/glm_mlx.py`

**Config:**
```python
@dataclass
class GLMWayfinderConfig:
    path: Literal["sparse", "permute"] = "permute"
    strategy: Literal["random", "greedy", "online_insertion", "regular_partition"] = "random"
    window: int = 64
    landmark_stride: Optional[int] = 64
    num_cycles: int = 1
    edge_disjoint: bool = True
    regular_num_clusters: int = 8
    seed: int = 0
    edge_bias: bool = True
    window_drop: float = 0.0
    compiled_graph_dir: Optional[str] = None
    permute_head_chunk_size: int = 2            # smaller than Qwen (MLA has large head dim)
    query_chunk_size: int = 192                 # smaller than Qwen
    permute_prepermute_mode: Literal["auto", "off", "kv", "qkv", "on"] = "auto"
    permute_log_chunks: bool = False
    compute_edge_utilization_proxy: bool = True
    compute_graph_metrics: bool = True
    retro_backfill_enabled: bool = False
    retro_backfill_alpha: float = 0.0
    retro_backfill_training_only: bool = True
    retro_backfill_causal_only: bool = True
    permute_memory_budget_bytes: Optional[int] = None
    active_dense_threshold: Optional[int] = None
    verify_spectral_gap: bool = False
    spectral_gap_threshold: float = 4.0
```

**GLM-specific MLA handling:**

GLM-4.7-Flash uses Multi-head Latent Attention (MLA) with compressed KV cache:
- `qk_dim = kv_lora_rank + qk_rope_head_dim` (Q/K dimension)
- `value_dim = kv_lora_rank` (V dimension, smaller than qk_dim)
- `n_kv_heads = 1` (MQA-style compressed KV)

**Critical:** Wayfinder kernels expect `dh_q == dh_k == dh_v`. GLM's MLA has `dh_v < dh_qk`.
Solution: `_pad_value_dim(values, target_dim)` pads values with zeros to match Q/K dim before
attention, then slices back after: `y_latent = y[..., :value_dim]`.

**QKV extraction:**
```python
def extract_qkv_from_glm_attention(attn, x, *, cache=None):
    # Returns (queries [B, Hq, Tq, Dqk], keys [B, 1, Tk, Dqk], values [B, 1, Tk, Dv])
    # Handles: optional LoRA Q path (q_a_proj -> q_a_layernorm -> q_b_proj)
    # MLA: kv_a_proj_with_mqa -> split compressed_kv + k_pe -> layernorm -> RoPE
    # embed_q projects Q nope portion
```

**Wrapper class:** `GLMWayfinderAttention(nn.Module)`

Key differences from Qwen:
- `_adaptive_graph_seq_len(*, k_len, q_len, cache)` — builds graph at adaptive horizon
  `max(k_len, cache.max_size)` to amortize rebuilds during chunked prefill
- Active mode: when `permute` path + cache exists + `q_len <= k_len`, uses
  `wayfinder_permute_window_attention_active_batched()` instead of dense fallback
- `force_dense_active` when `k_len <= active_dense_threshold`
- MLA output pipeline: `y[..., :value_dim]` -> `unembed_out()` -> `o_proj()`

**Swap function:**
```python
def swap_glm_attention_with_wayfinder(
    model: nn.Module,
    *,
    cfg: GLMWayfinderConfig,
    layer_indices: Optional[List[int]] = None,
) -> List[int]:
    # Handles both model.layers and model.model.layers patterns
```

### 6.3 Shared Graph Runtime

**File:** `hcsa/integrations/qwen_mlx.py` (class `_QwenGraphRuntime`, used by both Qwen and GLM)

```python
class _QwenGraphRuntime:
    def __init__(self, *, n_heads, window, landmark_stride, strategy, num_cycles,
                 edge_disjoint, regular_num_clusters, seed, path,
                 compiled_graph_dir, verify_spectral_gap, spectral_gap_threshold,
                 store_numpy_abi, store_graph_tensors): ...

    def get_or_build_cache(self, owner_id: int, T: int) -> tuple[_QwenGraphCache, bool]:
        # Two-level cache:
        #   1. Per-instance: _QWEN_GRAPH_CACHE_STORE[owner_id]
        #   2. By-key: _QWEN_GRAPH_CACHE_BY_KEY[cache_key]
        # Falls through to _load_compiled_cache (disk) then _build_graph_abi (runtime)
        ...
```

**Cache structure:**
```python
@dataclass(frozen=True)
class _QwenGraphCache:
    mlx_graph: MLXGraphABI
    numpy_abi: Optional[WayfinderGraphABI]
    safe_idx: mx.array              # [H, T, D] for sparse path
    causal_mask: mx.array           # [H, T, D] for sparse path
    perm_mx: List[mx.array]         # per-head primary perm
    inv_perm: List[mx.array]        # per-head inverse perm
    perm_mx_stacked: mx.array       # [H, T] or [H, d, T]
    inv_perm_stacked: mx.array      # [H, T] or [H, d, T]
    cache_key: tuple
    source: str = "runtime"
    artifact_dir: Optional[str] = None
    persistent_bytes: int = 0
```

### 6.4 GQA Helper

```python
def _repeat_kv_to_q_heads(x: mx.array, n_q_heads: int) -> mx.array:
    # [B, Hkv, T, dh] -> [B, Hq, T, dh] via broadcast + reshape
    # Used by sparse gather path (permute path handles GQA internally)
```

---

## 7. ZMLX Patch System Reference

### 7.1 Pattern Protocol

**File:** `zmlx/patch/_types.py`

```python
@runtime_checkable
class PatchPattern(Protocol):
    @property
    def name(self) -> str: ...

    def matches(self, module: Any, name: str, parent: Any | None = None) -> bool: ...

    def apply(self, module: Any, config: PatchConfig) -> Any: ...

@dataclass(frozen=True)
class PatchConfig:
    compute_dtype: str = "float32"
    threadgroup: int | str = 256
    verbose: bool = False
    moe_fused_swiglu_max_tokens: int | None = None
```

### 7.2 Registration

**File:** `zmlx/patch/_registry.py`

```python
_PATTERNS: dict[str, PatchPattern] = {}

def register(pattern: PatchPattern) -> PatchPattern:
    _PATTERNS[pattern.name] = pattern
    return pattern
```

New patterns self-register at import time. `_ensure_loaded()` imports all pattern modules from
`zmlx/patch/patterns/` on first access.

### 7.3 Traversal

**File:** `zmlx/patch/_traversal.py`

`apply_patterns(model, patterns, config)` does depth-first traversal of `model.children()`.
For each child, tries each pattern in order; first match wins. On match, calls
`pattern.apply(child, config)` and replaces the child on the parent via `setattr`.

### 7.4 Existing Patterns

| Pattern | File | Matches on | Replaces |
|---|---|---|---|
| `moe_mlp` | `moe_mlp.py` | MoE blocks (gate + experts) | `__call__` with fused gating/combine |
| `swiglu_mlp` | `swiglu_mlp.py` | Dense SwiGLU MLPs | `__call__` with fused kernel |
| `geglu_mlp` | `geglu_mlp.py` | Dense GeGLU MLPs | `__call__` with fused kernel |
| `softmax` | `softmax.py` | Attention modules with `.softmax` | `.softmax` attribute |
| `rmsnorm` | `rmsnorm.py` | RMSNorm layers | `__call__` with fused kernel |
| `layernorm` | `layernorm.py` | LayerNorm layers | `__call__` with fused kernel |
| `residual_norm` | `residual_norm.py` | Residual + norm pairs | Fused residual add + norm |
| `deepseek_router` | `deepseek_router.py` | DeepSeek expert routing | Optimized routing |
| `glm47_rope` | `glm47_rope.py` | GLM-4.7 RoPE | Optimized RoPE |

### 7.5 Safety Excludes

| Family | Fidelity excludes (always) | Perf excludes (unless custom primitives) |
|---|---|---|
| `qwen` | `swiglu_mlp`, `residual_norm` | `moe_mlp` |
| `gpt_oss` | `residual_norm` | — |
| `mixtral` | `moe_mlp` | — |
| `glm` | `rmsnorm` | `moe_mlp`, `swiglu_mlp` |

---

## 8. Integration Approaches

### 8.1 Approach A: External Composition (recommended first step)

Apply ZMLX and HCSA as separate post-load steps. No code changes to either repo.

```python
import mlx.core as mx
from mlx_lm.utils import load

import zmlx
from hcsa.integrations.glm_mlx import (
    GLMWayfinderConfig,
    swap_glm_attention_with_wayfinder,
)

# Load model
model, tokenizer = load("mlx-community/GLM-4.7-Flash-4bit")

# Step 1: ZMLX patches MoE blocks
zmlx.patch(model, profile="glm")

# Step 2: HCSA swaps attention layers
cfg = GLMWayfinderConfig(
    path="permute",
    strategy="random",
    window=64,
    num_cycles=1,
    seed=42,
    permute_head_chunk_size=2,
    query_chunk_size=192,
)
swapped = swap_glm_attention_with_wayfinder(model, cfg=cfg)
print(f"HCSA swapped {len(swapped)} attention layers")

# Now use model normally — MoE decode is fused, attention is sparse
```

Same pattern for Qwen3:
```python
from hcsa.integrations.qwen_mlx import (
    QwenWayfinderConfig,
    swap_qwen_attention_with_wayfinder,
)

model, tokenizer = load("mlx-community/Qwen3-30B-A3B-4bit")
zmlx.patch(model, profile="qwen3")
cfg = QwenWayfinderConfig(path="permute", strategy="random", window=64)
swap_qwen_attention_with_wayfinder(model, cfg=cfg)
```

### 8.2 Approach B: HCSA as a ZMLX Patch Pattern (future integration)

Add `sparse_attention.py` to `zmlx/patch/patterns/` following the existing pattern protocol.

**Sketch:**

```python
# zmlx/patch/patterns/sparse_attention.py
from __future__ import annotations
from typing import Any

from .._registry import register
from .._types import PatchConfig

# Defer HCSA imports to apply() to avoid hard dependency
_HCSA_AVAILABLE = None

def _check_hcsa():
    global _HCSA_AVAILABLE
    if _HCSA_AVAILABLE is None:
        try:
            import hcsa  # noqa: F401
            _HCSA_AVAILABLE = True
        except ImportError:
            _HCSA_AVAILABLE = False
    return _HCSA_AVAILABLE


class _SparseAttentionPattern:
    @property
    def name(self) -> str:
        return "sparse_attention"

    def matches(self, module: Any, name: str, parent: Any | None = None) -> bool:
        if name != "self_attn":
            return False
        if not _check_hcsa():
            return False
        # Detect supported model families
        mod_path = type(module).__module__ or ""
        cls_name = type(module).__name__
        if "qwen3" in mod_path or "Qwen3" in cls_name:
            return True
        if "glm" in mod_path or "GLM" in cls_name:
            return True
        return False

    def apply(self, module: Any, config: PatchConfig) -> Any:
        mod_path = type(module).__module__ or ""

        if "glm" in mod_path:
            from hcsa.integrations.glm_mlx import (
                GLMWayfinderAttention,
                GLMWayfinderConfig,
            )
            hcsa_cfg = GLMWayfinderConfig(
                path="permute",
                strategy="random",
                window=64,
                permute_head_chunk_size=2,
                query_chunk_size=192,
            )
            return GLMWayfinderAttention(module, hcsa_cfg)

        if "qwen3" in mod_path:
            from hcsa.integrations.qwen_mlx import (
                QwenWayfinderAttention,
                QwenWayfinderConfig,
            )
            hcsa_cfg = QwenWayfinderConfig(
                path="permute",
                strategy="random",
                window=64,
            )
            return QwenWayfinderAttention(module, hcsa_cfg)

        return module  # no-op fallback


register(_SparseAttentionPattern())
```

**Registration:** add to `_ensure_loaded()` in `_registry.py`:
```python
from .patterns import sparse_attention  # noqa: F401
```

**Safety excludes:** add to `__init__.py`:
```python
_PERF_EXCLUDES["glm"].add("sparse_attention")  # until chunked prefill bug is fixed
```

**Usage after integration:**
```python
zmlx.patch(model, patterns=["moe_mlp", "sparse_attention"])
```

### 8.3 Approach C: Paged Attention + Sparse Neighbor Indexing

ZMLX has a paged attention Metal kernel (`zmlx/kernels/attention.py:paged_attention`).
For long-context decode, HCSA's neighbor index could restrict which KV pages are fetched.

**Current paged attention signature:**
```python
def paged_attention(
    q,              # [B, H, D]
    k_cache,        # [N_BLOCKS, BS, HKV, D]
    v_cache,        # [N_BLOCKS, BS, HKV, D]
    block_table,    # [B, MAX_BLOCKS]
    context_lens,   # [B]
    scale=None,
    *,
    threadgroup=256,
    max_context=4096,
) -> mx.array:
```

**Sparse variant concept:**
Instead of scanning all `context_len / BS` blocks, use the graph ABI's neighbor index
to identify which blocks contain neighbors, fetch only those. This reduces block reads
from O(T/BS) to O(W/BS) at long contexts.

**Required changes:**
1. Build graph ABI for current sequence length
2. Extract unique block IDs from `neigh_idx // block_size`
3. Build sparse block table with only relevant blocks
4. Modify Metal kernel to iterate over sparse block list

**Feasibility:** Medium effort. The current Metal kernel uses a simple loop over all blocks.
Modifying it to use a sparse block list is straightforward Metal code. The graph ABI provides
the neighbor indices. Main challenge: the graph must be rebuilt or cached as context grows
during decode. For static strategies (random), the graph is cacheable.

---

## 9. Data Flow: End-to-End

### Full prefill path (GLM-4.7-Flash, HCSA permute, T=65536):

```
Input tokens [B, T]
    |
    v
Embedding + position encoding
    |
    v
For each of 47 transformer layers:
    |
    +-- input_layernorm(x)     [ZMLX: rmsnorm pattern if enabled]
    |
    +-- GLMWayfinderAttention.__call__(x_norm)
    |     |
    |     +-- extract_qkv_from_glm_attention()
    |     |     Q: [B, 32, T, 192]  (32 heads, qk_dim=192)
    |     |     K: [B, 1, T, 192]   (MQA, kv_lora_rank + rope_dim)
    |     |     V: [B, 1, T, 128]   (kv_lora_rank only)
    |     |
    |     +-- _pad_value_dim(V, 192)  -> V_padded: [B, 1, T, 192]
    |     |
    |     +-- _repeat_kv_to_q_heads(K, 32), _repeat_kv_to_q_heads(V_padded, 32)
    |     |
    |     +-- _QwenGraphRuntime.get_or_build_cache(T=65536)
    |     |     -> builds random Hamiltonian cycle per head
    |     |     -> caches perm/inv_perm [32, 65536]
    |     |     -> ~150ms first call, 0ms cached
    |     |
    |     +-- wayfinder_permute_window_attention_batched(
    |     |       Q, K_expanded, V_expanded,
    |     |       all_perms=[32, 65536],
    |     |       window=64,
    |     |       head_chunk_size=2,
    |     |       query_chunk_size=192,
    |     |   )
    |     |     -> O(T * 129) instead of O(T^2)
    |     |     -> output: [B, 32, T, 192]
    |     |
    |     +-- y[..., :128]  -> slice back to value_dim
    |     +-- unembed_out() -> o_proj()
    |     +-- output: [B, T, D]
    |
    +-- residual add
    |
    +-- post_attention_layernorm(x)
    |
    +-- SparseMoeBlock.__call__(x_norm)    [ZMLX: moe_mlp pattern]
    |     |
    |     +-- fused topk_gating_softmax()   [ZMLX Metal kernel]
    |     +-- expert SwiGLU                  [ZMLX fused or stock]
    |     +-- fused expert combine           [ZMLX Metal kernel]
    |     +-- shared_experts()
    |
    +-- residual add
    |
    v
Next layer
```

### Decode path (single token, T_cache=65536):

```
Input token [B, 1]
    |
    v
For each layer:
    +-- GLMWayfinderAttention.__call__(x)
    |     Q_len=1, K_len=T_cache
    |     -> dense fallback: scaled_dot_product_attention
    |     -> O(T_cache) — same as stock, no HCSA overhead
    |
    +-- SparseMoeBlock.__call__(x)
    |     -> ZMLX fused gating + combine    [the actual decode speedup]
```

---

## 10. Benchmarking Combined Pipeline

### Quick validation script

```python
#!/usr/bin/env python3
"""Benchmark ZMLX + HCSA combined on GLM-4.7-Flash-4bit."""
import time
import mlx.core as mx
from mlx_lm.utils import load, generate_step

import zmlx
from hcsa.integrations.glm_mlx import (
    GLMWayfinderConfig,
    swap_glm_attention_with_wayfinder,
)

MODEL_ID = "mlx-community/GLM-4.7-Flash-4bit"
PROMPT = "The theory of sparse attention mechanisms in transformer architectures"
MAX_TOKENS = 100

def bench(label, model, tokenizer):
    tokens = tokenizer.encode(PROMPT)
    input_ids = mx.array([tokens])

    # Prefill
    t0 = time.perf_counter()
    mx.eval(model(input_ids))
    prefill_ms = (time.perf_counter() - t0) * 1000

    # Decode
    t0 = time.perf_counter()
    count = 0
    for token, _ in zip(generate_step(input_ids, model), range(MAX_TOKENS)):
        mx.eval(token)
        count += 1
    decode_ms = (time.perf_counter() - t0) * 1000

    print(f"[{label}] prefill={prefill_ms:.0f}ms  "
          f"decode={decode_ms:.0f}ms ({count/(decode_ms/1000):.1f} tok/s)")

# Baseline
model, tokenizer = load(MODEL_ID)
bench("baseline", model, tokenizer)

# ZMLX only
model, tokenizer = load(MODEL_ID)
zmlx.patch(model, profile="glm")
bench("zmlx_only", model, tokenizer)

# HCSA only
model, tokenizer = load(MODEL_ID)
cfg = GLMWayfinderConfig(path="permute", strategy="random", window=64)
swap_glm_attention_with_wayfinder(model, cfg=cfg)
bench("hcsa_only", model, tokenizer)

# Combined
model, tokenizer = load(MODEL_ID)
zmlx.patch(model, profile="glm")
swap_glm_attention_with_wayfinder(model, cfg=cfg)
bench("combined", model, tokenizer)
```

### What to measure

| Metric | Baseline | +ZMLX | +HCSA | +Both |
|---|---|---|---|---|
| Prefill latency (ms) | O(T^2) | O(T^2) | O(T*W) | O(T*W) |
| Decode tok/s | baseline | +6-8% | ~same | +6-8% |
| Peak memory (GB) | baseline | ~same | -26% | -26% |
| Output quality | reference | identical | approximate | approximate |

**Expected outcome:** ZMLX decode speedup + HCSA prefill memory savings stack independently.
Decode tok/s improvement comes from ZMLX; memory reduction comes from HCSA. Neither interferes.

---

## 11. Adding HCSA Support for New ZMLX Models

To add HCSA sparse attention for a model that ZMLX already patches (e.g., LFM2):

### Step 1: Understand the attention module

```python
# Find the attention class
model, _ = load("mlx-community/LFM2-8B-A1B-4bit")
layer = model.layers[0]  # or model.model.layers[0]
attn = layer.self_attn
print(type(attn))          # e.g., mlx_lm.models.lfm2.Attention
print(dir(attn))           # find q_proj, k_proj, v_proj, o_proj, rope, etc.
print(attn.num_heads)      # query heads
print(attn.num_kv_heads)   # kv heads (for GQA)
```

### Step 2: Write QKV extraction function

```python
def extract_qkv_from_lfm2_attention(attn, x, *, cache=None):
    """Extract post-RoPE Q/K/V from LFM2 attention module."""
    B, L, _ = x.shape
    q = attn.q_proj(x).reshape(B, L, attn.num_heads, -1).transpose(0, 2, 1, 3)
    k = attn.k_proj(x).reshape(B, L, attn.num_kv_heads, -1).transpose(0, 2, 1, 3)
    v = attn.v_proj(x).reshape(B, L, attn.num_kv_heads, -1).transpose(0, 2, 1, 3)

    if cache is not None:
        q = attn.rope(q, offset=cache.offset)
        k = attn.rope(k, offset=cache.offset)
        k, v = cache.update_and_fetch(k, v)
    else:
        q = attn.rope(q)
        k = attn.rope(k)

    return q, k, v  # [B, Hq, T, dh], [B, Hkv, T, dh], [B, Hkv, T, dh]
```

### Step 3: Write wrapper attention class

Follow the `QwenWayfinderAttention` pattern:
- Copy projections from base module
- Create `_QwenGraphRuntime` (reusable for any model)
- Route to sparse or permute path based on config
- Handle GQA via `_repeat_kv_to_q_heads` (sparse path) or internal GQA mapping (permute path)
- Dense fallback for Q_len != K_len

### Step 4: Write swap function

```python
def swap_lfm2_attention_with_wayfinder(model, *, cfg, layer_indices=None):
    layers = model.layers  # or model.model.layers
    swapped = []
    indices = layer_indices or range(len(layers))
    for i in indices:
        base_attn = layers[i].self_attn
        layers[i].self_attn = LFM2WayfinderAttention(base_attn, cfg)
        swapped.append(i)
    return swapped
```

### Step 5: Verify correctness

```python
# Compare outputs between dense baseline and sparse attention
import numpy as np

model_dense, _ = load(MODEL_ID)
model_sparse, _ = load(MODEL_ID)
swap_lfm2_attention_with_wayfinder(model_sparse, cfg=cfg)

x = mx.random.normal((1, 128, model_dense.dims))
y_dense = model_dense(x)
y_sparse = model_sparse(x)

diff = float(mx.abs(y_dense - y_sparse).max())
print(f"Max absolute difference: {diff:.6f}")
# Expect small but nonzero (sparse approximation, not exact)
```

---

## 12. File Reference Index

### HCSA (wayfinder repo)

| File | Purpose |
|---|---|
| `hcsa/graph/abi.py` | `WayfinderGraphABI`, `EdgeType`, `build_graph_abi_from_adjacency`, `stack_head_abis`, `validate_graph_abi`, `graph_metrics` |
| `hcsa/mlx/graph_abi.py` | `MLXGraphABI`, `to_mlx_graph_abi`, `safe_neighbor_idx`, `causal_neighbor_mask` |
| `hcsa/graph_strategies.py` | `GraphStrategy` protocol, `RandomCycleStrategy`, `GreedyCycleStrategy`, `OnlineInsertionStrategy`, `RegularPartitionStrategy`, `build_strategy` |
| `hcsa/topology/core.py` | `Topology`, `TopologyGraph` — orchestrates per-head strategy + caching |
| `hcsa/cycles.py` | `random_cycle`, `greedy_cycle`, `online_insertion_step`, `edge_disjoint_random_cycles`, `regular_partition_cycle` |
| `hcsa/mlx/attention.py` | `dense_causal_attention`, `sparse_gather_attention`, `wayfinder_permute_window_attention_batched`, `wayfinder_permute_window_attention_active_batched`, `stable_masked_softmax`, `WayfinderAttentionMLX`, `AttentionProfile` |
| `hcsa/integrations/qwen_mlx.py` | `QwenWayfinderConfig`, `QwenWayfinderAttention`, `swap_qwen_attention_with_wayfinder`, `_QwenGraphRuntime`, `_QwenGraphCache`, `extract_qkv_from_qwen_attention`, `_repeat_kv_to_q_heads` |
| `hcsa/integrations/glm_mlx.py` | `GLMWayfinderConfig`, `GLMWayfinderAttention`, `swap_glm_attention_with_wayfinder`, `extract_qkv_from_glm_attention`, `_pad_value_dim` |

### ZMLX

| File | Purpose |
|---|---|
| `zmlx/patch/__init__.py` | `patch()`, `unpatch()`, `smart_patch()`, presets, safety excludes |
| `zmlx/patch/_types.py` | `PatchPattern` protocol, `PatchConfig`, `PatchResult` |
| `zmlx/patch/_registry.py` | `register()`, `get_pattern()`, `_ensure_loaded()` |
| `zmlx/patch/_traversal.py` | `apply_patterns()`, `_walk_modules()` |
| `zmlx/patch/patterns/moe_mlp.py` | MoE MLP pattern (gating + combine fusion) |
| `zmlx/patch/patterns/softmax.py` | Softmax pattern (Metal kernel replacement) |
| `zmlx/patch/patterns/swiglu_mlp.py` | Dense SwiGLU pattern (activation fusion) |
| `zmlx/kernels/attention.py` | `paged_attention`, `masked_softmax`, `scale_mask_softmax`, `logsumexp_lastdim` |
| `zmlx/kernels/moe.py` | `topk_gating_softmax`, expert combine kernels |
| `zmlx/kernels/transformer.py` | `swiglu2`, RMSNorm, LayerNorm, RoPE kernels |

---

## 13. Key Constraints and Gotchas

1. **HCSA is not numerically identical.** Sparse attention is an approximation. Quality depends
   on window size, strategy, and num_cycles. Always validate perplexity/quality on your task.

2. **Decode is unaffected by HCSA.** Q_len=1 attention is already O(T). HCSA falls back to dense
   for decode. All decode speedup comes from ZMLX's MoE fusion.

3. **Graph construction is CPU-bound.** For random strategy at T=65536, graph build takes ~150ms
   (one-time, cached). For greedy strategy, it's O(T^2) and requires routing embeddings.

4. **MLA requires value padding.** GLM's MLA has `dh_v < dh_qk`. The integration pads values
   to match Q/K dim before attention, then slices back. This adds a small constant overhead.

5. **GQA handling differs by path.** Sparse gather path needs `_repeat_kv_to_q_heads()` to
   broadcast KV. Permute path handles GQA internally via `q_to_kv_head` mapping.

6. **Chunked prefill active mode is GLM-only.** Qwen integration falls back to dense for
   Q_len != K_len. GLM integration has the `active_batched` path but it's the source of the
   latency regression bug (partially fixed).

7. **ZMLX `patch()` and HCSA `swap_*()` order doesn't matter** since they touch different
   module attributes. But if you use ZMLX's `unpatch()`, it won't know about HCSA's swaps
   (and vice versa). Track both separately.

8. **`python3` not `python`** on this system. All scripts should use `python3`.

9. **Memory budget.** HCSA's permute path supports `memory_budget_bytes` to cap peak memory
   during prepermute planning. Set this if running on constrained hardware (e.g., M1 Pro 16GB).

10. **Spectral gap verification** (`verify_spectral_gap=True`) adds ~50ms per graph build to
    validate Hamiltonian expansion quality. Useful for debugging, not recommended for production.
