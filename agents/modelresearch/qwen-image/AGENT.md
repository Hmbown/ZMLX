# Qwen-Image Advisor Agent (txt2img + edit)

Use `agents/modelresearch/TEMPLATE_AGENT.md` as the baseline, plus the Qwen-Image-specific notes to fill in:

## Evidence pointers
- Model cards: `exo/resources/image_model_cards/exolabs--Qwen-Image*.toml` (components: text encoder, transformer, VAE)
- Pipeline block configs: `exo/src/exo/worker/engines/image/models/qwen/config.py`
- txt2img adapter: `exo/src/exo/worker/engines/image/models/qwen/adapter.py`
- edit adapter: `exo/src/exo/worker/engines/image/models/qwen/edit_adapter.py`
- Block wrappers: `exo/src/exo/worker/engines/image/models/qwen/wrappers.py`

## Config snapshot (from bundled Exo model cards)
- Qwen-Image transformer: `n_layers = 60`
- Components: `text_encoder/` (12 layers, no sharding), `transformer/` (can shard), `vae/` (no sharding).

