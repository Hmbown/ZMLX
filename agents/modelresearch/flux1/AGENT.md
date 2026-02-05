# FLUX.1 Advisor Agent (image diffusion)

Use `agents/modelresearch/TEMPLATE_AGENT.md` as the baseline, plus the FLUX-specific notes to fill in:

## Evidence pointers
- Model cards: `exo/resources/image_model_cards/exolabs--FLUX.1-*.toml` (components: encoders, transformer, VAE)
- Pipeline block configs: `exo/src/exo/worker/engines/image/models/flux/config.py`
- Model adapter: `exo/src/exo/worker/engines/image/models/flux/adapter.py`
- Block wrappers (attention + RoPE handling): `exo/src/exo/worker/engines/image/models/flux/wrappers.py`

## Config snapshot (from bundled Exo model cards)
- Flux cards enumerate component sizes and whether `transformer/` can shard (`can_shard = true`).

