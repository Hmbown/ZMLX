# Next Session: Discover Autorun + Model Benchmarking

## What This Is

ZMLX Discover is an LLM-guided Metal kernel search system. It uses PUCT tree
search (AlphaZero-style) to find optimized GPU kernels for Apple Silicon.
An LLM generates Metal kernel variants, each is compiled, correctness-checked
against a Python reference, and benchmarked on the actual GPU. The best kernels
can replace ZMLX's current hand-written kernels if they're faster.

**This session's job**: Run Discover at scale, benchmark winners on real models,
and integrate improvements into ZMLX's default kernel set.

## Prerequisites

```bash
cd /Volumes/VIXinSSD/ZMLX
source .venv/bin/activate
```

No API keys needed — the `claude-code` backend shells out to the local `claude`
CLI with `--dangerously-skip-permissions`.

## Step 1: Run Full Autorun

Run all targets with thorough search (10 steps, 4 candidates/step = 40
candidates per target):

```bash
python -m zmlx.discover autorun \
  --backends claude-code \
  --steps 10 \
  --candidates-per-step 4 \
  --verbose
```

This runs all 9 targets: `fused_swiglu`, `rmsnorm`, `moe_combine`,
`topk_gating`, `ttt_linear_decode`, plus GLM-specific variants of the first 4.

Expected time: ~3-5 minutes per target (LLM generation + GPU benchmarking).
Total: ~30-45 minutes.

After completion, compare results:

```bash
python -m zmlx.discover compare discover_sessions/autorun
```

Record which targets achieved >1.05x speedup. Likely winners based on early
results (3 steps, 2 cands):
- `glm_fused_swiglu`: 1.37x (2-wide ILP)
- `glm_rmsnorm`: 1.12x (float8 vectorized)

## Step 2: Export Winning Kernels

For each target with >1.05x speedup:

```bash
python -m zmlx.discover export \
  discover_sessions/autorun/<target>_claude-code/<target>_session.json \
  -o src/zmlx/kernels/discovered/<target>.py
```

## Step 3: Benchmark on Real Models

This is the critical validation step. Micro-kernel speedups don't always
translate to model throughput gains (kernel may not be on the hot path, or
dispatch overhead may negate the gain at decode batch sizes).

Test each model in three configurations:

### 3a. Baseline MLX (no ZMLX)

```bash
python -m zmlx.validate mlx-community/GLM-4-9B-Chat-4bit \
  --max-tokens 200 --runs 5 --no-patch
```

### 3b. Current ZMLX

```bash
python -m zmlx.validate mlx-community/GLM-4-9B-Chat-4bit \
  --max-tokens 200 --runs 5
```

### 3c. ZMLX with Discovered Kernels

To test discovered kernels, temporarily wire them into the patch patterns.
The discovered kernel files in `src/zmlx/kernels/discovered/` are ready-to-use
Python modules. To integrate:

1. In the relevant patch pattern (e.g., `src/zmlx/patch/patterns/swiglu_mlp.py`),
   import the discovered kernel and use it instead of the current one
2. Re-run validation

### 3d. Record Results

Create `discover_sessions/benchmark_results.json`:

```json
[
  {
    "model": "mlx-community/GLM-4-9B-Chat-4bit",
    "baseline_tps": 0.0,
    "current_zmlx_tps": 0.0,
    "discovered_zmlx_tps": 0.0,
    "discovered_delta_pct": 0.0,
    "token_identical": true
  }
]
```

## Step 4: Integration Decision

For each target where model-level gains are confirmed:

1. Move the discovered kernel from `kernels/discovered/` to the main catalog
2. Update the patch pattern to use it
3. Add correctness test to `tests/test_kernels_catalog.py`
4. Run full test suite: `pytest -q`
5. Run fidelity validation: `python -m zmlx.validate <model> --max-tokens 500`
6. Update CHANGELOG.md

For targets with NO model-level gains:
- Keep the session data for reference
- Note that the kernel-level speedup didn't translate
- The kernel may be useful for different batch sizes or future models

## Step 5: Optional Multi-Backend Comparison

If API keys are available, compare LLM backends:

```bash
export ANTHROPIC_API_KEY=...  # for claude backend
export OPENAI_API_KEY=...     # for openai backend

python -m zmlx.discover autorun \
  --targets <winning_targets_only> \
  --backends "claude-code,claude,openai" \
  --steps 10 \
  --candidates-per-step 4 \
  --session-dir discover_sessions/multi_backend \
  --verbose

python -m zmlx.discover compare discover_sessions/multi_backend
```

## Step 6: Add New Model Targets

If adding targets for other models (LFM2, Qwen3), add target functions to
`src/zmlx/discover/targets.py`. Key dimensions to look up:

| Model | hidden_size | expert_intermediate | num_experts | top_k |
|:--|--:|--:|--:|--:|
| GLM-4.7-Flash | 2048 | 1536 | 64 | 4 |
| LFM2-8B-A1B | 4096 | 2048 | 128 | 2 |
| Qwen3-30B-A3B | 2048 | 1536 | 128 | 8 |

Pattern for new targets:

```python
def lfm_fused_swiglu_target() -> SearchSpace:
    """Fused SwiGLU for LFM2 MoE experts: D=2048."""
    return fused_swiglu_target(D=2048)

# Register in TARGETS dict at bottom of file
```

## Current State (2026-02-07)

- Branch: `ttt-discover`
- Discover module: `src/zmlx/discover/` (fully functional)
- Autorun + compare commands: working
- Claude Code backend: working (no API key needed)
- Early results: glm_fused_swiglu 1.37x, glm_rmsnorm 1.12x (3 steps only)
- TTT module: `src/zmlx/ttt/` (experimental, not related to main Discover flow)
- Unstaged files: discover/, ttt/, tests, benchmarks notes

## Key Files

| File | Purpose |
|:--|:--|
| `src/zmlx/discover/__main__.py` | CLI: search, autorun, compare, report, export |
| `src/zmlx/discover/targets.py` | All target definitions with model dimensions |
| `src/zmlx/discover/llm.py` | LLM backends (claude-code, claude, openai, mock) |
| `src/zmlx/discover/evaluate.py` | Compile + correctness + benchmark pipeline |
| `src/zmlx/discover/tree.py` | PUCT tree search |
| `src/zmlx/discover/prompts.py` | System/user prompts, response parsing |
| `src/zmlx/kernels/discovered/` | Auto-generated kernel exports |
| `docs/DISCOVER_PLAYBOOK.md` | Full reference with architecture notes |
| `discover_sessions/` | Saved search sessions |
