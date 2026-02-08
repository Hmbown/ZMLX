#!/usr/bin/env bash
# Create (or update) a local MLX checkout at ./mlx_local and apply the
# gather_qmm_swiglu patch, then optionally build the Python extension.
#
# The ./mlx_local directory is gitignored by ZMLX; this script is meant for
# local developer/user setup.
#
# Usage:
#   bash integrations/mlx_local_integration/setup_mlx_local.sh
#
# Options:
#   --no-build          Apply patch but skip build
#   --python <path>     Python interpreter to use for build (default: auto-pick)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MLX_DIR="${REPO_ROOT}/mlx_local"
MLX_REPO_URL="${MLX_REPO_URL:-https://github.com/ml-explore/mlx.git}"
MLX_REF="${MLX_REF:-185b06d9efc1c869540eccfb5baff853fff3659d}"
PATCH_FILE="${SCRIPT_DIR}/gather_qmm_swiglu.patch"

NO_BUILD=0
PYTHON=""

die() {
  echo "ERROR: $*" >&2
  exit 1
}

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

pick_python() {
  for py in python3.14 python3.13 python3; do
    if has_cmd "$py"; then
      echo "$py"
      return 0
    fi
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build)
      NO_BUILD=1
      shift
      ;;
    --python)
      [[ $# -ge 2 ]] || die "--python requires a value"
      PYTHON="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--no-build] [--python <path>]"
      exit 0
      ;;
    *)
      die "Unknown arg: $1"
      ;;
  esac
done

has_cmd git || die "git not found (required)"
[[ -f "${PATCH_FILE}" ]] || die "Missing patch file: integrations/mlx_local_integration/gather_qmm_swiglu.patch"

if [[ -z "${PYTHON}" ]]; then
  PYTHON="$(pick_python || true)"
fi
[[ -n "${PYTHON}" ]] || die "No python interpreter found"

echo "=== Custom MLX setup (gather_qmm_swiglu) ==="
echo "Repo root: <REPO_ROOT>"
echo "MLX dir:   ./mlx_local"
echo "MLX ref:   ${MLX_REF}"
echo "Python:    $("${PYTHON}" --version 2>&1)"
echo ""

if [[ ! -d "${MLX_DIR}/.git" ]]; then
  echo "Cloning MLX into ./mlx_local..."
  git clone "${MLX_REPO_URL}" "${MLX_DIR}"
fi

if git -C "${MLX_DIR}" diff --quiet && git -C "${MLX_DIR}" diff --cached --quiet; then
  echo "Checking out MLX ref: ${MLX_REF}"
  git -C "${MLX_DIR}" fetch --quiet --all --tags || true
  git -C "${MLX_DIR}" checkout --quiet "${MLX_REF}" || die "Failed to checkout MLX ref: ${MLX_REF}"
else
  echo "NOTE: ./mlx_local has local changes; skipping checkout to ${MLX_REF}"
fi

echo "Applying MLX patch (idempotent)..."
if git -C "${MLX_DIR}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "  Patch already applied."
elif git -C "${MLX_DIR}" apply --check "${PATCH_FILE}" >/dev/null 2>&1; then
  git -C "${MLX_DIR}" apply "${PATCH_FILE}"
  echo "  Patch applied."
else
  die "MLX patch does not apply cleanly. Ensure MLX is at ${MLX_REF} (or a compatible commit)."
fi

if [[ "${NO_BUILD}" == "1" ]]; then
  echo ""
  echo "Skipping build (--no-build)."
  echo "To build later:"
  echo "  cd mlx_local && ${PYTHON} setup.py build_ext --inplace"
  exit 0
fi

echo ""
echo "Building MLX Python extension (in-place)..."
(
  cd "${MLX_DIR}"
  "${PYTHON}" setup.py build_ext --inplace
)

echo ""
echo "=== Verifying ==="
HAS_FUSED="$(
  PYTHONPATH="${MLX_DIR}/python" "${PYTHON}" -c "import mlx.core as mx; print(hasattr(mx, 'gather_qmm_swiglu'))" 2>&1 || true
)"
echo "  gather_qmm_swiglu: ${HAS_FUSED}"

if [[ "${HAS_FUSED}" != "True" ]]; then
  echo ""
  echo "WARNING: gather_qmm_swiglu not available after build."
  echo "  Verify your build output and ensure the patch applied."
fi

echo ""
echo "=== Done ==="
echo "Next: run `bash setup_zmlx.sh` to wire exo to ZMLX (and optionally ./mlx_local)."
