# Contributing to ZMLX

## Setup
ZMLX targets macOS on Apple Silicon (M-series) with MLX installed.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install mlx
pip install -e ".[dev]"
```

## Tests
```bash
pytest -q
```

Note: Most tests require Metal on macOS arm64. On unsupported hosts,
tests are skipped (see `tests/conftest.py`).

## Lint / type-check (optional)
```bash
ruff check .
mypy src
```

## Adding a kernel
1. Pick the right module under `src/zmlx/kernels/` or add a new one.
2. Follow existing patterns:
   - cache kernels with `@cache` and validate shapes early
   - accept `threadgroup` when relevant (all kernels compute in float32 internally)
   - use `DEFAULT_HEADER` for shared helpers (sigmoid/silu/gelu_tanh)
3. Add or update `__all__` in the module.
4. Add a correctness test in `tests/` (compare against MLX reference ops).
5. Update `docs/KERNELS.md` and, if user-facing, `README.md`.
6. Add an example under `examples/` for new public APIs.

## Project conventions
- Keep public APIs small and documented.
- Prefer simple, explicit kernel sources over heavy DSL magic.
- Always add correctness tests vs MLX reference ops.
- For performance work, include a reproducible benchmark script and report settings (shape, dtype, device).

## PRs
- Keep PRs focused.
- Update `README.md` if user-facing behavior changes.
- If adding new public API, add at least one example in `examples/`.
 - Ensure `docs/KERNELS.md` matches the actual kernel catalog.
