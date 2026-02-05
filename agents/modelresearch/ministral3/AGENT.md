# Ministral3 (Mistral-family) Advisor Agent

Use `agents/modelresearch/TEMPLATE_AGENT.md` as the baseline. This file records
what is currently verifiable in this repo.

## Evidence pointers
- Sharding: `exo/src/exo/worker/engines/mlx/auto_parallel.py` treats `Ministral3Model` using `LlamaShardingStrategy`.
- If model cards are added later, extend `agents/modelresearch/INVENTORY.md` and add a config snapshot section.

## Bundled model IDs (from Exo model cards)

- None yet. Exo imports `mlx_lm.models.ministral3.Model` for sharding, but there are currently no `exo/resources/inference_model_cards/*.toml` entries for Ministral3.

## Exo integration notes (verified in this repo)

- Tensor sharding: `exo/src/exo/worker/engines/mlx/auto_parallel.py`
  - Exo imports `mlx_lm.models.ministral3.Model as Ministral3Model`.
  - `tensor_auto_parallel()` routes `Ministral3Model` through `LlamaShardingStrategy`, implying a Llama-like attention + MLP surface.

## Verified module surface (from Exo sharding code)

Because Ministral3 is treated as Llama-like for sharding, Exo implicitly requires:
- Attention: `layer.self_attn.{q_proj,k_proj,v_proj,o_proj}`, `n_heads`, and optional `n_kv_heads`.
- MLP: `layer.mlp.{gate_proj,up_proj,down_proj}`.
