#!/usr/bin/env bash
# Setup ZMLX integration for exo (optional GLM/Qwen3 MoE decode acceleration).
#
# This script clones exo into ./exo (gitignored), applies the ZMLX hook patch,
# creates an exo venv, installs exo + ZMLX, and generates an `exo/run_zmlx.sh`
# launcher.
#
# Optional: if a custom MLX build exists at `./mlx_local/python`, it is added to
# the exo venv via a `.pth` file so `mx.gather_qmm_swiglu` is available.
#
# Prerequisites:
#   - macOS on Apple Silicon (for MLX Metal backend)
#   - Python >= 3.13 (exo requires >= 3.13)
#   - git
#   - uv (recommended) — `brew install uv`
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXO_DIR="${REPO_ROOT}/exo"
EXO_REPO_URL="${EXO_REPO_URL:-https://github.com/exo-explore/exo.git}"
# Pin to a known-good commit for the patch. Override with EXO_REF=main if desired.
EXO_REF="${EXO_REF:-a0f4f363555744f2a9660679437be064bb2bb712}"

EXO_PATCH="${REPO_ROOT}/integrations/exo_integration/exo_zmlx.patch"

MLX_LOCAL_PY="${REPO_ROOT}/mlx_local/python"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

pick_python() {
  for py in python3.14 python3.13 python3; do
    if ! has_cmd "$py"; then
      continue
    fi
    if "$py" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 13) else 1)'; then
      echo "$py"
      return 0
    fi
  done
  return 1
}

echo "=== ZMLX + exo setup ==="
echo "Repo root: <REPO_ROOT>"
echo ""

has_cmd git || die "git not found (required)"
has_cmd uv || die "uv not found. Install with: brew install uv"

PYTHON="$(pick_python || true)"
if [[ -z "${PYTHON}" ]]; then
  die "Python >= 3.13 not found. Install Python 3.13+ (exo requires >= 3.13)."
fi
echo "Using Python: $("$PYTHON" --version)"
echo ""

if [[ ! -f "${EXO_PATCH}" ]]; then
  die "Missing patch file: integrations/exo_integration/exo_zmlx.patch"
fi

if [[ ! -d "${EXO_DIR}/.git" ]]; then
  echo "Cloning exo into ./exo..."
  git clone "${EXO_REPO_URL}" "${EXO_DIR}"
fi

if git -C "${EXO_DIR}" diff --quiet && git -C "${EXO_DIR}" diff --cached --quiet; then
  echo "Checking out exo ref: ${EXO_REF}"
  git -C "${EXO_DIR}" fetch --quiet --all --tags || true
  git -C "${EXO_DIR}" checkout --quiet "${EXO_REF}" || die "Failed to checkout exo ref: ${EXO_REF}"
else
  echo "NOTE: ./exo has local changes; skipping checkout to ${EXO_REF}"
fi

echo "Applying exo patch (idempotent)..."
if git -C "${EXO_DIR}" apply --reverse --check "${EXO_PATCH}" >/dev/null 2>&1; then
  echo "  Patch already applied."
elif git -C "${EXO_DIR}" apply --check "${EXO_PATCH}" >/dev/null 2>&1; then
  git -C "${EXO_DIR}" apply "${EXO_PATCH}"
  echo "  Patch applied."
else
  die "exo patch does not apply cleanly. Try setting EXO_REF=main (or a compatible commit)."
fi

VENV="${EXO_DIR}/.venv"
if [[ -d "${VENV}" ]]; then
  echo "Removing existing exo venv..."
  rm -rf "${VENV}"
fi

echo "Creating exo venv..."
uv venv -p "${PYTHON}" "${VENV}"

echo "Installing exo + ZMLX (editable)..."
uv pip install -e "${EXO_DIR}" --python "${VENV}/bin/python"
uv pip install -e "${REPO_ROOT}" --python "${VENV}/bin/python"

SITE_PACKAGES="$("${VENV}/bin/python" -c "import site; print(site.getsitepackages()[0])")"
PTH_FILE="${SITE_PACKAGES}/zmlx_mlx_local.pth"

CUSTOM_SO="$(ls "${MLX_LOCAL_PY}"/mlx/core*.so 2>/dev/null | head -n 1 || true)"
if [[ -n "${CUSTOM_SO}" ]]; then
  echo "Found custom MLX build: ${CUSTOM_SO#${REPO_ROOT}/}"
  echo "Wiring custom MLX into exo venv via .pth..."
  echo "${MLX_LOCAL_PY}" > "${PTH_FILE}"

  if ! "${VENV}/bin/python" -c "import mlx.core as mx" >/dev/null 2>&1; then
    echo "WARNING: Custom MLX failed to import in the exo venv; removing .pth."
    rm -f "${PTH_FILE}"
    CUSTOM_SO=""
  fi
else
  echo "Custom MLX not found at ./mlx_local/python (optional)."
  echo "  For GLM/Qwen3 gains, follow docs/EXPERIMENTAL_MLX.md to build a custom MLX."
fi

echo ""
echo "=== Verifying ==="
ZMLX_VER="$("${VENV}/bin/python" -c "import zmlx; print(zmlx.__version__)" 2>&1 || true)"
MLX_VER="$("${VENV}/bin/python" -c "import mlx.core as mx; print(getattr(mx, '__version__', 'unknown'))" 2>&1 || true)"
HAS_FUSED="$("${VENV}/bin/python" -c "import mlx.core as mx; print(hasattr(mx, 'gather_qmm_swiglu'))" 2>&1 || true)"

echo "  ZMLX:              ${ZMLX_VER}"
echo "  MLX:               ${MLX_VER}"
echo "  gather_qmm_swiglu: ${HAS_FUSED}"

if [[ -n "${CUSTOM_SO}" && "${HAS_FUSED}" != "True" ]]; then
  echo ""
  echo "WARNING: Custom MLX path is present but gather_qmm_swiglu is unavailable."
  echo "  GLM/Qwen3 MoE fusions will be auto-skipped."
fi

RUN_SCRIPT="${EXO_DIR}/run_zmlx.sh"
cat > "${RUN_SCRIPT}" <<'LAUNCHER'
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate

export EXO_ZMLX="${EXO_ZMLX:-1}"
export EXO_ZMLX_VERBOSE="${EXO_ZMLX_VERBOSE:-1}"

python -m exo --api-port "${EXO_API_PORT:-52416}"
LAUNCHER
chmod +x "${RUN_SCRIPT}"

echo ""
echo "=== Done ==="
echo ""
echo "Launch exo with ZMLX:"
echo "  bash exo/run_zmlx.sh"
echo ""
echo "Expected log on model load:"
echo "  [zmlx] Applying fused-kernel patches ..."
