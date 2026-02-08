# ZMLX Discover: Automated Kernel Search Playbook

This document is both a reference and an **executable prompt** for AI sessions.
When starting a new session, paste the relevant phase as your instruction.

## Overview

Discover uses LLM-guided PUCT tree search to find optimized Metal kernels for
Apple Silicon. The workflow:

1. **Search**: LLM generates Metal kernel variants for each target
2. **Evaluate**: Each variant is compiled, correctness-checked, and benchmarked
3. **Compare**: Best kernels from each backend/target are compared
4. **Benchmark**: Discovered kernels are tested on actual models vs baseline
5. **Integrate**: Winners replace current ZMLX kernels if they beat existing perf

## Quick Reference

```bash
# Activate venv first
source .venv/bin/activate

# List available targets
python -m zmlx.discover list

# Single target search
python -m zmlx.discover search glm_fused_swiglu --llm claude-code --steps 10 -v

# Run all targets with one backend
python -m zmlx.discover autorun --backends claude-code --steps 10

# Run all targets across multiple backends
python -m zmlx.discover autorun --backends "claude-code,claude,openai" --steps 10

# Run only GLM targets
python -m zmlx.discover autorun --targets glm_fused_swiglu glm_rmsnorm glm_moe_combine glm_topk_gating

# Compare results across backends
python -m zmlx.discover compare discover_sessions/autorun

# View a specific session
python -m zmlx.discover report discover_sessions/autorun/glm_fused_swiglu_claude-code/glm_fused_swiglu_session.json

# Export best kernel
python -m zmlx.discover export <session.json> -o <output.py>
```

## Targets

Current targets with real model dimensions:

| Target | Kernel | Dimensions | Model Family |
|:--|:--|:--|:--|
| `glm_fused_swiglu` | SwiGLU activation | D=1536 | GLM-4.7-Flash |
| `glm_rmsnorm` | RMSNorm | D=2048 | GLM-4.7-Flash |
| `glm_moe_combine` | Expert output combine | D=2048, K=4 | GLM-4.7-Flash |
| `glm_topk_gating` | Top-K routing | D=64, K=4 | GLM-4.7-Flash |
| `fused_swiglu` | SwiGLU activation | D=4096 | Generic |
| `rmsnorm` | RMSNorm | D=4096 | Generic |
| `moe_combine` | Expert output combine | D=4096, K=2 | Generic |
| `topk_gating` | Top-K routing | D=8, K=2 | Generic |
| `ttt_linear_decode` | TTT-Linear decode | F=64 | TTT (experimental) |

### Adding New Targets

To add model-specific targets (e.g., for LFM2 or Qwen3), add functions to
`src/zmlx/discover/targets.py` following the pattern of `glm_*_target()`.
Key dimensions to look up per model:

- **hidden_size**: Main hidden dim (used in rmsnorm, combine)
- **expert_intermediate_size**: FFN inner dim (used in swiglu)
- **num_experts / top_k**: MoE routing params (used in gating, combine)

## LLM Backends

| Backend | CLI Flag | Auth Required | Notes |
|:--|:--|:--|:--|
| `claude-code` | `--llm claude-code` | None (uses local `claude` CLI) | **Recommended default** |
| `claude` | `--llm claude` | `ANTHROPIC_API_KEY` | Direct API, faster |
| `openai` | `--llm openai` | `OPENAI_API_KEY` | GPT-4o |
| `mock` | `--llm mock` | None | Testing only, trivial mutations |

## Phase 1: Full Autorun (AI Session Prompt)

```
Run ZMLX Discover autorun across all targets. Use the claude-code backend
(no API key needed). Run 10 steps with 4 candidates per step for thorough
coverage. Use verbose mode.

Commands:
  source .venv/bin/activate
  python -m zmlx.discover autorun \
    --backends claude-code \
    --steps 10 \
    --candidates-per-step 4 \
    --verbose

After completion, run:
  python -m zmlx.discover compare discover_sessions/autorun

Report the comparison table and note which targets achieved >1.05x speedup.
```

## Phase 2: Multi-Backend Comparison (AI Session Prompt)

```
Run ZMLX Discover with multiple LLM backends to compare quality.
Prerequisites: Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY env vars.

For targets that showed >1.05x with claude-code, re-run with all available
backends to find the best kernel across models:

  source .venv/bin/activate
  python -m zmlx.discover autorun \
    --targets <winning_targets_from_phase1> \
    --backends "claude-code,claude,openai" \
    --steps 10 \
    --candidates-per-step 4 \
    --session-dir discover_sessions/multi_backend \
    --verbose

Then compare:
  python -m zmlx.discover compare discover_sessions/multi_backend

Export the overall best kernel for each target:
  python -m zmlx.discover export <best_session.json> -o src/zmlx/kernels/discovered/<target>.py
```

## Phase 3: Model-Level Benchmarking (AI Session Prompt)

```
Benchmark discovered kernels against baseline MLX and current ZMLX on actual
models. This validates that micro-kernel speedups translate to real model
throughput gains.

For each model, run three configurations:
1. Baseline (no ZMLX): python -m zmlx.validate <model> --max-tokens 200 --runs 5 --no-patch
2. Current ZMLX: python -m zmlx.validate <model> --max-tokens 200 --runs 5
3. ZMLX + Discovered: [integrate discovered kernels first, then validate]

Models to test:
- mlx-community/GLM-4-9B-Chat-4bit (or GLM-4.7-Flash equivalent)
- mlx-community/LFM2-8B-A1B-4bit
- mlx-community/Qwen3-30B-A3B-4bit (if custom MLX available)

Record results in discover_sessions/benchmark_results.json with format:
{
  "model": "...",
  "baseline_tps": 0.0,
  "current_zmlx_tps": 0.0,
  "discovered_zmlx_tps": 0.0,
  "discovered_delta_pct": 0.0,
  "token_identical": true
}

The key question: does the discovered kernel's micro-benchmark speedup
(e.g., 1.37x on glm_fused_swiglu) translate to measurable model-level
throughput improvement? Even a 1-2% model-level gain is worth keeping if
it's token-identical.
```

## Phase 4: Integration Decision (AI Session Prompt)

```
Review discover_sessions/benchmark_results.json from Phase 3.

For each target where discovered kernels show model-level gains:

1. Export the best kernel:
   python -m zmlx.discover export <session.json> -o src/zmlx/kernels/discovered/<target>.py

2. Create or update the corresponding patch pattern in src/zmlx/patch/patterns/
   to use the discovered kernel when available.

3. Add the discovered kernel to the test suite (tests/test_kernels_catalog.py)
   with correctness tests against the MLX reference.

4. Run the full test suite:
   pytest -q

5. Run fidelity validation on affected models:
   python -m zmlx.validate <model> --max-tokens 500 --runs 5

6. If token-identical and faster, update CHANGELOG.md and create a PR.

For targets where discovered kernels show NO model-level gains:
- Keep the session data for reference
- The kernel-level speedup didn't translate (likely not on the hot path)
- Note this in the session for future investigation
```

## Architecture Notes

### How Discover Works

1. **Target definition** (`targets.py`): Defines the kernel signature, reference
   implementation, test shapes, grid computation, and optimization constraints.

2. **LLM generation** (`llm.py`, `prompts.py`): Sends the current best kernel +
   history to an LLM, asking for N variants with different optimization strategies.

3. **Evaluation** (`evaluate.py`): Compiles each variant via `mx.fast.metal_kernel`,
   checks correctness against the Python reference, then benchmarks latency.

4. **PUCT tree search** (`tree.py`): Selects which node to expand next using
   Upper Confidence bounds (like AlphaZero). Balances exploring new strategies
   vs exploiting known-good ones.

5. **Session persistence** (`session.py`): Auto-saves after each step. Sessions
   can be resumed with `--resume`.

### Key Design Decisions

- **Seed kernel always included**: The seed (baseline) kernel gets evaluated first,
  so we know the kernel-level baseline (not just the MLX reference).

- **N constant for bounds checking**: Elementwise targets define `constexpr uint N`
  so LLM-generated vectorized variants can do bounds checking.

- **Grid oversizing is safe**: Grid is set to N total threads for elementwise
  kernels. Vectorized variants that process K elements per thread have (N-K)
  threads hit the `if (idx >= N) return;` guard — wasteful but safe.

- **ClaudeCodeBackend**: No API key needed. Shells out to `claude -p --model sonnet
  --dangerously-skip-permissions`. Slower than direct API but zero setup.

### Common Failure Modes

| Failure | Cause | Fix |
|:--|:--|:--|
| "Unable to build metal library" | MSL syntax error or undefined symbol | Check that all constants (N, D, K, etc.) are defined in seed |
| Correctness mismatch | Kernel produces wrong results | LLM generated buggy logic; handled by eval pipeline |
| Timeout | Kernel hangs or takes too long | `--timeout` flag; kernel safety checks for infinite loops |
| 0 candidates parsed | LLM response doesn't match expected format | Check `parse_llm_response` regex patterns |

## Session File Format

```json
{
  "schema_version": "1.0",
  "metadata": {
    "session_id": "hex16",
    "target_name": "glm_fused_swiglu",
    "llm_backend": "claude-code",
    "device_chip": "Apple M4",
    "device_memory_gb": 36,
    "baseline_us": 141.3,
    "best_speedup": 1.37,
    "best_reward": 1.35,
    "best_source": "...",
    "total_steps": 3,
    "total_candidates": 6,
    "total_evaluated": 6
  },
  "tree_data": { ... },
  "candidate_sources": { "gen1_abc123": "...", ... }
}
```

## Results So Far (2026-02-07, M4 Mac Studio 36GB)

| Target | Backend | Steps | Speedup | Strategy |
|:--|:--|--:|--:|:--|
| `glm_fused_swiglu` | claude-code | 3 | **1.37x** | 2-wide ILP-interleaved |
| `glm_rmsnorm` | claude-code | 3 | **1.12x** | float8 vectorized loads |

These are micro-benchmark results (kernel-level). Model-level validation pending.
