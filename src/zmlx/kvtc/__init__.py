"""KVTC — KV Cache Transform Coding for MLX models.

Compresses LLM KV caches using PCA decorrelation, DP-optimal mixed-precision
quantization, and zlib entropy coding.  Achieves up to 20x compression vs FP16
with configurable quality.

Supported models:
  - LFM2-8B-A1B  (dual_stream, 24 layers, 8 KV heads, head_dim=64)
  - Qwen3-30B-A3B (dual_stream, 48 layers, 4 KV heads, head_dim=128)
  - GLM-4.7-Flash (single_stream MLA, 47 layers, 1 head, head_dim=576)

Quick start::

    from zmlx.kvtc import KVTCCacheCodec, CalibrationArtifacts, model_preset

    preset = model_preset("lfm2")
    arts = CalibrationArtifacts.from_dir("calibration/lfm2/")
    codec = KVTCCacheCodec(arts, rope_cfg=preset.rope, mode=preset.mode)

    blob = codec.compress(k_layers, v_layers)
    k_hat, v_hat = codec.decompress(blob)
"""

from .codec import CalibrationArtifacts, KVTCCacheCodec
from .plan import GroupSpec, QuantPlan
from .presets import KVTCPreset, list_presets, model_preset
from .rope import RotaryConfig, RotaryEmbedding

__all__ = [
    "CalibrationArtifacts",
    "GroupSpec",
    "KVTCCacheCodec",
    "KVTCPreset",
    "QuantPlan",
    "RotaryConfig",
    "RotaryEmbedding",
    "list_presets",
    "model_preset",
]
