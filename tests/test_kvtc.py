"""Tests for the KVTC subpackage.

Runs synthetic roundtrip tests for all 3 model geometries (LFM2, Qwen3, GLM),
plus preset value assertions, RoPE traditional + offset roundtrip, and bitpack
roundtrip.  No model loading required — all data is random.
"""

from __future__ import annotations

import numpy as np
import pytest

from zmlx.kvtc.bitpack import pack_uint, packed_nbytes, unpack_uint
from zmlx.kvtc.calibration.dp_calibrate import calibrate_dp_plan
from zmlx.kvtc.codec import CalibrationArtifacts, KVTCCacheCodec
from zmlx.kvtc.plan import GroupSpec, QuantPlan
from zmlx.kvtc.presets import list_presets, model_preset
from zmlx.kvtc.rope import RotaryConfig, RotaryEmbedding

# ── Bitpack tests ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_bitpack_roundtrip(bits):
    rng = np.random.default_rng(42)
    values = rng.integers(0, (1 << bits), size=100, dtype=np.uint8)
    packed = pack_uint(values, bits)
    assert len(packed) == packed_nbytes(100, bits)
    unpacked, consumed = unpack_uint(packed, bits, 100)
    np.testing.assert_array_equal(unpacked, values)
    assert consumed == len(packed)


def test_bitpack_zero_bits():
    packed = pack_uint(np.array([1, 2, 3], dtype=np.uint8), 0)
    assert packed == b""
    unpacked, consumed = unpack_uint(b"", 0, 5)
    assert consumed == 0
    assert len(unpacked) == 5


# ── Plan tests ────────────────────────────────────────────────────────────


def test_plan_json_roundtrip():
    plan = QuantPlan(groups=[
        GroupSpec(size=16, qtype="int4"),
        GroupSpec(size=8, qtype="int2"),
        GroupSpec(size=4, qtype="fp8"),
    ])
    j = plan.to_json()
    plan2 = QuantPlan.from_json(j)
    assert plan == plan2
    assert plan.r() == 28


# ── RoPE tests ────────────────────────────────────────────────────────────


def test_rope_interleaved_roundtrip():
    """Standard interleaved RoPE: apply then invert should recover original."""
    cfg = RotaryConfig(dim=64, base=10000.0, traditional=False, offset=0)
    rope = RotaryEmbedding(cfg)
    rng = np.random.default_rng(123)
    x = rng.standard_normal((2, 10, 64)).astype(np.float32)  # (heads, seq, dim)
    positions = np.arange(10, dtype=np.int64)

    rotated = rope.apply(x, positions, inverse=False)
    recovered = rope.apply(rotated, positions, inverse=True)
    np.testing.assert_allclose(recovered, x, atol=1e-5)


def test_rope_traditional_roundtrip():
    """Half-split (traditional) RoPE roundtrip."""
    cfg = RotaryConfig(dim=64, base=1_000_000.0, traditional=True, offset=0)
    rope = RotaryEmbedding(cfg)
    rng = np.random.default_rng(456)
    x = rng.standard_normal((1, 8, 64)).astype(np.float32)
    positions = np.arange(8, dtype=np.int64)

    rotated = rope.apply(x, positions, inverse=False)
    recovered = rope.apply(rotated, positions, inverse=True)
    np.testing.assert_allclose(recovered, x, atol=1e-5)


def test_rope_offset_roundtrip():
    """RoPE with offset (GLM MLA style): only last 64 dims rotated."""
    cfg = RotaryConfig(dim=64, base=1_000_000.0, traditional=True, offset=512)
    rope = RotaryEmbedding(cfg)
    rng = np.random.default_rng(789)
    x = rng.standard_normal((1, 8, 576)).astype(np.float32)  # (heads, seq, head_dim)
    positions = np.arange(8, dtype=np.int64)

    rotated = rope.apply(x, positions, inverse=False)
    # First 512 dims should be unchanged
    np.testing.assert_array_equal(rotated[..., :512], x[..., :512])
    # Last 64 dims should differ (unless all zeros)
    assert not np.allclose(rotated[..., 512:], x[..., 512:])

    recovered = rope.apply(rotated, positions, inverse=True)
    np.testing.assert_allclose(recovered, x, atol=1e-5)


def test_rope_partial_rotary():
    """RoPE with dim < head_dim, no offset (standard partial rotary)."""
    cfg = RotaryConfig(dim=32, base=10000.0, traditional=False, offset=0)
    rope = RotaryEmbedding(cfg)
    rng = np.random.default_rng(111)
    x = rng.standard_normal((2, 6, 64)).astype(np.float32)
    positions = np.arange(6, dtype=np.int64)

    rotated = rope.apply(x, positions, inverse=False)
    # Last 32 dims should be unchanged
    np.testing.assert_array_equal(rotated[..., 32:], x[..., 32:])

    recovered = rope.apply(rotated, positions, inverse=True)
    np.testing.assert_allclose(recovered, x, atol=1e-5)


# ── Preset tests ──────────────────────────────────────────────────────────


def test_preset_lfm2():
    p = model_preset("lfm2")
    assert p.layers == 24
    assert p.kv_heads == 8
    assert p.head_dim == 64
    assert p.mode == "dual_stream"
    assert p.rope.dim == 64
    assert p.rope.traditional is False
    assert p.rope.offset == 0


def test_preset_qwen3():
    p = model_preset("qwen3")
    assert p.layers == 48
    assert p.kv_heads == 4
    assert p.head_dim == 128
    assert p.mode == "dual_stream"
    assert p.rope.dim == 128
    assert p.rope.traditional is False
    assert p.rope.offset == 0


def test_preset_glm():
    p = model_preset("glm")
    assert p.layers == 47
    assert p.kv_heads == 1
    assert p.head_dim == 576
    assert p.mode == "single_stream"
    assert p.v_dim == 0
    assert p.rope.dim == 64
    assert p.rope.traditional is True
    assert p.rope.offset == 512


def test_preset_fuzzy_match():
    assert model_preset("lfm2-8b").name == "lfm2"
    assert model_preset("GLM-4.7-Flash").name == "glm"
    assert model_preset("qwen3-30b-a3b").name == "qwen3"


def test_preset_list():
    presets = list_presets()
    names = {p.name for p in presets}
    assert names == {"lfm2", "qwen3", "glm"}


def test_preset_unknown():
    with pytest.raises(KeyError, match="Unknown preset"):
        model_preset("nonexistent-model")


# ── Synthetic roundtrip helpers ───────────────────────────────────────────


def _make_synthetic_artifacts(
    L: int, H: int, D: int, r: int = 8, n_cal: int = 64,
) -> CalibrationArtifacts:
    """Create synthetic calibration artifacts from random data."""
    rng = np.random.default_rng(42)
    p = L * H * D

    # Random calibration data
    C = rng.standard_normal((n_cal, p)).astype(np.float32)

    from zmlx.kvtc.calibration.dp_calibrate import calibrate_dp_plan
    basis, plan = calibrate_dp_plan(C, max_bit_budget=p * 4, r=r)

    return CalibrationArtifacts(
        k_mu=basis.mu.astype(np.float16),
        k_V=basis.V.astype(np.float16),
        k_plan=plan,
        v_mu=basis.mu.astype(np.float16),
        v_V=basis.V.astype(np.float16),
        v_plan=plan,
        meta={"synthetic": True},
    )


def _make_single_stream_artifacts(
    L: int, H: int, D: int, r: int = 8, n_cal: int = 64,
) -> CalibrationArtifacts:
    """Create synthetic calibration artifacts for single_stream (keys only)."""
    rng = np.random.default_rng(42)
    p = L * H * D

    C = rng.standard_normal((n_cal, p)).astype(np.float32)

    from zmlx.kvtc.calibration.dp_calibrate import calibrate_dp_plan
    basis, plan = calibrate_dp_plan(C, max_bit_budget=p * 4, r=r)

    # Dummy V artifacts for single_stream
    return CalibrationArtifacts(
        k_mu=basis.mu.astype(np.float16),
        k_V=basis.V.astype(np.float16),
        k_plan=plan,
        v_mu=np.zeros(1, dtype=np.float16),
        v_V=np.zeros((1, 1), dtype=np.float16),
        v_plan=QuantPlan(groups=[]),
        meta={"synthetic": True, "mode": "single_stream"},
    )


# ── Dual-stream roundtrip tests ──────────────────────────────────────────


class TestDualStreamRoundtrip:
    """Synthetic roundtrip for dual_stream models (LFM2, Qwen3 geometry)."""

    @pytest.fixture(params=[
        # (L, H, D, seq, label)
        (2, 2, 8, 64, "tiny"),
        (24, 8, 64, 200, "lfm2-like"),
    ], ids=lambda x: x[-1])
    def geometry(self, request):
        return request.param

    def test_roundtrip(self, geometry):
        L, H, D, seq, _label = geometry
        rng = np.random.default_rng(0)

        k_layers = [rng.standard_normal((1, H, seq, D)).astype(np.float16) for _ in range(L)]
        v_layers = [rng.standard_normal((1, H, seq, D)).astype(np.float16) for _ in range(L)]

        arts = _make_synthetic_artifacts(L, H, D)
        codec = KVTCCacheCodec(arts, w=4, s=2, mode="dual_stream")

        blob = codec.compress(k_layers, v_layers)
        k_hat, v_hat = codec.decompress(blob)

        assert len(k_hat) == L
        assert len(v_hat) == L

        for layer_idx in range(L):
            # Prefix/suffix should be exactly preserved
            np.testing.assert_array_equal(
                k_hat[layer_idx][:, :, :2, :], k_layers[layer_idx][:, :, :2, :]
            )
            np.testing.assert_array_equal(
                k_hat[layer_idx][:, :, -4:, :], k_layers[layer_idx][:, :, -4:, :]
            )
            np.testing.assert_array_equal(
                v_hat[layer_idx][:, :, :2, :], v_layers[layer_idx][:, :, :2, :]
            )
            np.testing.assert_array_equal(
                v_hat[layer_idx][:, :, -4:, :], v_layers[layer_idx][:, :, -4:, :]
            )

            # Middle region: lossy but bounded
            mid_k_orig = k_layers[layer_idx][:, :, 2:-4, :].astype(np.float32)
            mid_k_hat = k_hat[layer_idx][:, :, 2:-4, :].astype(np.float32)
            mse_k = float(np.mean((mid_k_orig - mid_k_hat) ** 2))
            assert mse_k < 10.0, f"K MSE too high: {mse_k}"


class TestDualStreamQwen3Geometry:
    """Roundtrip with Qwen3-like geometry (fewer heads, larger dim)."""

    def test_roundtrip(self):
        L, H, D, seq = 4, 4, 16, 80
        rng = np.random.default_rng(1)

        k_layers = [rng.standard_normal((1, H, seq, D)).astype(np.float16) for _ in range(L)]
        v_layers = [rng.standard_normal((1, H, seq, D)).astype(np.float16) for _ in range(L)]

        arts = _make_synthetic_artifacts(L, H, D)
        codec = KVTCCacheCodec(arts, w=4, s=2, mode="dual_stream")

        blob = codec.compress(k_layers, v_layers)
        k_hat, v_hat = codec.decompress(blob)

        assert len(k_hat) == L
        for layer_idx in range(L):
            np.testing.assert_array_equal(
                k_hat[layer_idx][:, :, :2, :], k_layers[layer_idx][:, :, :2, :]
            )


# ── Single-stream roundtrip tests (GLM MLA) ──────────────────────────────


class TestSingleStreamRoundtrip:
    """Synthetic roundtrip for single_stream (GLM MLA geometry)."""

    def test_roundtrip_glm_geometry(self):
        # Scaled-down GLM: 4 layers, 1 head, head_dim=32 (instead of 576)
        L, H, D, seq = 4, 1, 32, 64
        rng = np.random.default_rng(2)

        k_layers = [rng.standard_normal((1, H, seq, D)).astype(np.float16) for _ in range(L)]
        # V layers are empty for single_stream
        v_layers = [np.zeros((1, H, seq, 0), dtype=np.float16) for _ in range(L)]

        arts = _make_single_stream_artifacts(L, H, D)
        codec = KVTCCacheCodec(arts, w=4, s=2, mode="single_stream")

        blob = codec.compress(k_layers, v_layers)
        k_hat, v_hat = codec.decompress(blob)

        assert len(k_hat) == L
        assert len(v_hat) == L

        # V layers should all be zero with dim=0
        for layer_idx in range(L):
            assert v_hat[layer_idx].shape[-1] == 0

        # K prefix/suffix exact match
        for layer_idx in range(L):
            np.testing.assert_array_equal(
                k_hat[layer_idx][:, :, :2, :], k_layers[layer_idx][:, :, :2, :]
            )
            np.testing.assert_array_equal(
                k_hat[layer_idx][:, :, -4:, :], k_layers[layer_idx][:, :, -4:, :]
            )

            # Middle region: lossy
            mid_orig = k_layers[layer_idx][:, :, 2:-4, :].astype(np.float32)
            mid_hat = k_hat[layer_idx][:, :, 2:-4, :].astype(np.float32)
            mse = float(np.mean((mid_orig - mid_hat) ** 2))
            assert mse < 10.0, f"K MSE too high: {mse}"

    def test_blob_metadata_records_mode(self):
        """Verify that the compressed blob metadata contains mode info."""
        import json
        import struct

        L, H, D, seq = 2, 1, 16, 32
        rng = np.random.default_rng(3)

        k_layers = [rng.standard_normal((1, H, seq, D)).astype(np.float16) for _ in range(L)]
        v_layers = [np.zeros((1, H, seq, 0), dtype=np.float16) for _ in range(L)]

        arts = _make_single_stream_artifacts(L, H, D)
        codec = KVTCCacheCodec(arts, w=4, s=2, mode="single_stream")
        blob = codec.compress(k_layers, v_layers)

        # Parse header
        hdr_size = struct.calcsize("<8sHHI")
        _magic, _ver, _res, meta_len = struct.unpack_from("<8sHHI", blob, 0)
        meta = json.loads(blob[hdr_size : hdr_size + meta_len].decode("utf-8"))
        assert meta["mode"] == "single_stream"


# ── Roundtrip with RoPE ──────────────────────────────────────────────────


class TestRoPERoundtrip:
    """Roundtrip with RoPE enabled (unrotate on compress, rerotate on decompress)."""

    def test_roundtrip_with_rope(self):
        L, H, D, seq = 2, 2, 8, 32
        rng = np.random.default_rng(4)

        k_layers = [rng.standard_normal((1, H, seq, D)).astype(np.float16) for _ in range(L)]
        v_layers = [rng.standard_normal((1, H, seq, D)).astype(np.float16) for _ in range(L)]

        arts = _make_synthetic_artifacts(L, H, D)
        rope_cfg = RotaryConfig(dim=8, base=10000.0)
        codec = KVTCCacheCodec(
            arts, w=4, s=2,
            rope_cfg=rope_cfg, apply_rope_to_keys=True,
            mode="dual_stream",
        )

        blob = codec.compress(k_layers, v_layers)
        k_hat, v_hat = codec.decompress(blob)

        for layer_idx in range(L):
            # Prefix/suffix exact
            np.testing.assert_array_equal(
                k_hat[layer_idx][:, :, :2, :], k_layers[layer_idx][:, :, :2, :]
            )


# ── Calibration DP test ───────────────────────────────────────────────────


def test_calibrate_dp_produces_valid_plan():
    rng = np.random.default_rng(99)
    p = 32
    C = rng.standard_normal((50, p)).astype(np.float32)
    basis, plan = calibrate_dp_plan(C, max_bit_budget=p * 4, r=16)

    assert basis.mu.shape == (p,)
    assert basis.V.shape[0] == p
    assert basis.V.shape[1] == plan.r()
    assert plan.r() > 0
    assert plan.r() <= 16


# ── Import / CLI smoke tests ─────────────────────────────────────────────


def test_import_kvtc():
    from zmlx.kvtc import CalibrationArtifacts, KVTCCacheCodec, model_preset
    assert KVTCCacheCodec is not None
    assert CalibrationArtifacts is not None
    assert callable(model_preset)


def test_import_via_zmlx():
    import zmlx
    kvtc = zmlx.kvtc
    assert hasattr(kvtc, "KVTCCacheCodec")
    assert hasattr(kvtc, "model_preset")
