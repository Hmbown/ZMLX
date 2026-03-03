#!/usr/bin/env bash
# ============================================================================
# Isolation-first benchmark campaign (updated 2026-02-13).
#
# This replaces the older multi-variant single-process workflow that could OOM
# on GLM/Qwen3-sized models.
#
# Usage:
#   bash benchmarks/run_3hr_benchmark_campaign.sh <phase>
#
# Phases:
#   quick               - quick isolation checks at 200 tokens
#   glm_abba_200        - GLM AB/BA consistency block at 200 tokens
#   glm_abba_1024       - GLM AB/BA consistency block at 1024 tokens
#   qwen_isolation_200  - Qwen isolation matrix at 200 tokens
#   qwen_isolation_1024 - Qwen isolation matrix at 1024 tokens
#   glm_stress          - GLM prompt-diversity stress benchmark
#   matrix_controls     - matrix ledger control anchors (--patterns none)
#   checks              - ruff + pytest correctness gate
#   all                 - run every phase above, in order
#
# Tunables (env):
#   DATE=<YYYYMMDD>
#   RUNS_QUICK=<int>          (default: 3)
#   RUNS_MAIN=<int>           (default: 5)
#   SLEEP_BETWEEN_BLOCKS=<s>  (default: 45)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PHASE="${1:-}"
if [[ -z "$PHASE" ]]; then
  cat <<'USAGE'
Usage: bash benchmarks/run_3hr_benchmark_campaign.sh <phase>

Phases:
  quick
  glm_abba_200
  glm_abba_1024
  qwen_isolation_200
  qwen_isolation_1024
  glm_stress
  matrix_controls
  checks
  all
USAGE
  exit 1
fi

# Activate venv
source .venv/bin/activate

DATE="${DATE:-$(date +%Y%m%d)}"
RUNS_QUICK="${RUNS_QUICK:-3}"
RUNS_MAIN="${RUNS_MAIN:-5}"
SLEEP_BETWEEN_BLOCKS="${SLEEP_BETWEEN_BLOCKS:-45}"

CAPSULE_DIR="benchmarks/repro_capsules"
RESULTS_DIR="benchmarks/results"
SUMMARY="$RESULTS_DIR/campaign_${DATE}_${PHASE}_summary.txt"

mkdir -p "$CAPSULE_DIR" "$RESULTS_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SUMMARY"; }
hr()  { echo "========================================" | tee -a "$SUMMARY"; }

run_iso() {
  local suite="$1"
  local runs="$2"
  local max_tokens="$3"
  local prefix="$4"
  shift 4

  python benchmarks/bench_iso_variant_sweep.py \
    --suite "$suite" \
    --runs "$runs" \
    --max-tokens "$max_tokens" \
    --prefix "$prefix" \
    --variants "$@" \
    2>&1 | tee -a "$SUMMARY"
}

cooldown() {
  local seconds="$1"
  if [[ "$seconds" -gt 0 ]]; then
    log "Cooldown: sleeping ${seconds}s"
    sleep "$seconds"
  fi
}

run_block() {
  local label="$1"
  shift
  local start
  start=$(date +%s)
  log ">>> STARTING: $label"
  "$@"
  local elapsed=$(( $(date +%s) - start ))
  local min=$(( elapsed / 60 ))
  local sec=$(( elapsed % 60 ))
  log "<<< FINISHED: $label (${min}m ${sec}s)"
  hr
}

phase_quick() {
  run_iso glm47 "$RUNS_QUICK" 200 "glm47_quickval_iso_t200_r${RUNS_QUICK}_${DATE}" \
    control_swiglu_moe glm_combine_fp32_no_fma

  run_iso qwen3 "$RUNS_QUICK" 200 "qwen3_quickval_iso_t200_r${RUNS_QUICK}_${DATE}" \
    control_patterns_moe_mlp qwen_combine_exact qwen_router_argpartition_logits \
    qwen_router_argpartition_logits_topk_combine_exact
}

phase_glm_abba_200() {
  run_iso glm47 "$RUNS_MAIN" 200 "glm47_consistency_ab_t200_r${RUNS_MAIN}_${DATE}" \
    control_swiglu_moe glm_combine_fp32_no_fma
  cooldown "$SLEEP_BETWEEN_BLOCKS"
  run_iso glm47 "$RUNS_MAIN" 200 "glm47_consistency_ba_t200_r${RUNS_MAIN}_${DATE}" \
    glm_combine_fp32_no_fma control_swiglu_moe
}

phase_glm_abba_1024() {
  run_iso glm47 "$RUNS_MAIN" 1024 "glm47_consistency_ab_t1024_r${RUNS_MAIN}_${DATE}" \
    control_swiglu_moe glm_combine_fp32_no_fma
  cooldown "$SLEEP_BETWEEN_BLOCKS"
  run_iso glm47 "$RUNS_MAIN" 1024 "glm47_consistency_ba_t1024_r${RUNS_MAIN}_${DATE}" \
    glm_combine_fp32_no_fma control_swiglu_moe
}

phase_qwen_isolation_200() {
  run_iso qwen3 "$RUNS_MAIN" 200 "qwen3_isolation_ordered_t200_r${RUNS_MAIN}_repA_${DATE}" \
    control_patterns_moe_mlp qwen_combine_exact qwen_router_argpartition_logits \
    qwen_router_argpartition_logits_topk_combine_exact
  cooldown "$SLEEP_BETWEEN_BLOCKS"
  run_iso qwen3 "$RUNS_MAIN" 200 "qwen3_isolation_ordered_t200_r${RUNS_MAIN}_repB_${DATE}" \
    qwen_router_argpartition_logits_topk_combine_exact qwen_router_argpartition_logits \
    qwen_combine_exact control_patterns_moe_mlp
}

phase_qwen_isolation_1024() {
  run_iso qwen3 "$RUNS_MAIN" 1024 "qwen3_isolation_ordered_t1024_r${RUNS_MAIN}_repA_${DATE}" \
    control_patterns_moe_mlp qwen_combine_exact qwen_router_argpartition_logits \
    qwen_router_argpartition_logits_topk_combine_exact
  cooldown "$SLEEP_BETWEEN_BLOCKS"
  run_iso qwen3 "$RUNS_MAIN" 1024 "qwen3_isolation_ordered_t1024_r${RUNS_MAIN}_repB_${DATE}" \
    qwen_router_argpartition_logits_topk_combine_exact qwen_router_argpartition_logits \
    qwen_combine_exact control_patterns_moe_mlp
}

phase_glm_stress() {
  python benchmarks/bench_glm_stress.py \
    --prompts english_technical,chinese,code,math_reasoning,creative \
    --lengths 256,1024,2048 \
    --runs 3 \
    --warmup 2 \
    --json-out "$CAPSULE_DIR/glm47_stress_full_${DATE}.json" \
    --log-dir "$RESULTS_DIR/glm_stress" \
    2>&1 | tee -a "$SUMMARY"
}

phase_matrix_controls() {
  python -m zmlx.matrix run mlx-community/GLM-4.7-Flash-4bit-mxfp4 \
    --runs 3 --max-tokens 200 --patterns none \
    --notes "Isolation campaign control anchor (patterns none)" \
    2>&1 | tee -a "$SUMMARY"

  python -m zmlx.matrix run mlx-community/Qwen3-30B-A3B-4bit \
    --runs 3 --max-tokens 200 --patterns none \
    --notes "Isolation campaign control anchor (patterns none)" \
    2>&1 | tee -a "$SUMMARY"

  python -m zmlx.matrix run mlx-community/LFM2-8B-A1B-4bit \
    --runs 3 --max-tokens 200 --patterns none \
    --notes "Isolation campaign control anchor (patterns none)" \
    2>&1 | tee -a "$SUMMARY"
}

phase_checks() {
  ruff check . 2>&1 | tee -a "$SUMMARY"
  pytest -q 2>&1 | tee -a "$SUMMARY"
}

> "$SUMMARY"
hr
log "Isolation benchmark campaign"
log "Phase selector: $PHASE"
log "Hardware: $(sysctl -n machdep.cpu.brand_string)"
log "MLX: $(python -c 'import mlx.core as mx; print(mx.__version__)')"
log "ZMLX: $(python -c 'import zmlx; print(zmlx.__version__)')"
log "Custom primitive: $(python -c 'import mlx.core as mx; print(hasattr(mx, "gather_qmm_swiglu"))')"
log "Git: $(git rev-parse --short HEAD) on $(git branch --show-current)"
hr

case "$PHASE" in
  quick)
    run_block "Quick isolation (200 tok)" phase_quick
    ;;
  glm_abba_200)
    run_block "GLM AB/BA consistency (200 tok)" phase_glm_abba_200
    ;;
  glm_abba_1024)
    run_block "GLM AB/BA consistency (1024 tok)" phase_glm_abba_1024
    ;;
  qwen_isolation_200)
    run_block "Qwen isolation matrix (200 tok)" phase_qwen_isolation_200
    ;;
  qwen_isolation_1024)
    run_block "Qwen isolation matrix (1024 tok)" phase_qwen_isolation_1024
    ;;
  glm_stress)
    run_block "GLM stress benchmark" phase_glm_stress
    ;;
  matrix_controls)
    run_block "Matrix control anchors" phase_matrix_controls
    ;;
  checks)
    run_block "Repository checks (ruff + pytest)" phase_checks
    ;;
  all)
    run_block "Quick isolation (200 tok)" phase_quick
    run_block "GLM AB/BA consistency (200 tok)" phase_glm_abba_200
    run_block "GLM AB/BA consistency (1024 tok)" phase_glm_abba_1024
    run_block "Qwen isolation matrix (200 tok)" phase_qwen_isolation_200
    run_block "Qwen isolation matrix (1024 tok)" phase_qwen_isolation_1024
    run_block "GLM stress benchmark" phase_glm_stress
    run_block "Matrix control anchors" phase_matrix_controls
    run_block "Repository checks (ruff + pytest)" phase_checks
    ;;
  *)
    echo "Unknown phase: $PHASE" >&2
    exit 1
    ;;
esac

log "Campaign phase complete. Summary: $SUMMARY"
