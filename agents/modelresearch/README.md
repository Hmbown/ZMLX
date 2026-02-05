# Model Research Agents (Exo + ZMLX)

This folder contains “advisor agent” prompts for the model families currently supported by Exo’s local inference stack (MLX text models + MLX diffusion/image models).

Each `AGENT.md` is intended to be used as a system prompt (or pinned reference) for an internal “PhD-level” advisor focused on:
- architecture deep-dives,
- configuration + shape mapping,
- Exo integration constraints,
- and where ZMLX/MLX kernel work can realistically move the needle.

## Ground truth sources in this repo

- Model inventories (what Exo considers supported-by-default):
  - `exo/resources/inference_model_cards/*.toml`
  - `exo/resources/image_model_cards/*.toml`
- Exo MLX text engine integration:
  - `exo/src/exo/worker/engines/mlx/auto_parallel.py`
  - `exo/src/exo/worker/engines/mlx/utils_mlx.py`
  - `exo/src/exo/worker/runner/runner.py`
- Exo image engine integration:
  - `exo/src/exo/worker/engines/image/`

## How to use

- Pick a family folder (e.g. `deepseek/`) and use `AGENT.md` as the agent’s system prompt.
- When the agent makes performance claims, require a repro capsule under `benchmarks/repro_capsules/` (no unverifiable numbers).
- When referencing paths in writeups, use repo-relative paths or `<REPO_ROOT>/...` (never machine-specific absolute paths).

## Updating these agents

When Exo adds/removes model families, update:
- `agents/modelresearch/INVENTORY.md`
- Add/remove a family folder with an `AGENT.md`

