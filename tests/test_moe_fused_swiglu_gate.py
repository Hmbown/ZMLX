import mlx.core as mx
import pytest

from zmlx.patch.patterns import moe_mlp


@pytest.mark.cpu
def test_moe_fused_swiglu_token_gate():
    assert moe_mlp._should_fuse_swiglu_tokens(1, 1)
    assert not moe_mlp._should_fuse_swiglu_tokens(2, 1)


@pytest.mark.cpu
def test_qwen_fused_swiglu_default_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZMLX_QWEN_FUSED_SWIGLU", raising=False)
    assert not moe_mlp._qwen_allow_fused_swiglu()


@pytest.mark.cpu
def test_qwen_fused_swiglu_opt_out(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZMLX_QWEN_FUSED_SWIGLU", "0")
    assert not moe_mlp._qwen_allow_fused_swiglu()


@pytest.mark.cpu
def test_qwen_fused_swiglu_opt_in(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZMLX_QWEN_FUSED_SWIGLU", "1")
    assert moe_mlp._qwen_allow_fused_swiglu()


@pytest.mark.cpu
def test_qwen_fused_downproj_combine_default_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZMLX_QWEN_FUSED_DOWNPROJ_COMBINE", raising=False)
    assert not moe_mlp._qwen_allow_fused_downproj_combine()


@pytest.mark.cpu
def test_qwen_fused_downproj_combine_opt_in(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZMLX_QWEN_FUSED_DOWNPROJ_COMBINE", "1")
    assert moe_mlp._qwen_allow_fused_downproj_combine()


@pytest.mark.cpu
def test_qwen_fused_downproj_combine_kvec_default_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZMLX_QWEN_FUSED_DOWNPROJ_COMBINE_KVEC", raising=False)
    assert not moe_mlp._qwen_allow_fused_downproj_combine_kvec()


@pytest.mark.cpu
def test_qwen_fused_downproj_combine_kvec_opt_in(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZMLX_QWEN_FUSED_DOWNPROJ_COMBINE_KVEC", "1")
    assert moe_mlp._qwen_allow_fused_downproj_combine_kvec()


@pytest.mark.cpu
def test_qwen_fused_router_topk_default_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZMLX_QWEN_FUSED_ROUTER_TOPK", raising=False)
    assert not moe_mlp._qwen_allow_fused_router_topk()


@pytest.mark.cpu
def test_qwen_fused_router_topk_opt_in(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZMLX_QWEN_FUSED_ROUTER_TOPK", "1")
    assert moe_mlp._qwen_allow_fused_router_topk()


@pytest.mark.cpu
def test_qwen_router_argpartition_logits_default_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZMLX_QWEN_ROUTER_ARGPARTITION_LOGITS", raising=False)
    assert not moe_mlp._qwen_allow_logit_argpartition_router()


@pytest.mark.cpu
def test_qwen_router_argpartition_logits_opt_in(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZMLX_QWEN_ROUTER_ARGPARTITION_LOGITS", "1")
    assert moe_mlp._qwen_allow_logit_argpartition_router()


@pytest.mark.cpu
def test_qwen_router_argpartition_logits_topk_default_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZMLX_QWEN_ROUTER_ARGPARTITION_LOGITS_TOPK", raising=False)
    assert not moe_mlp._qwen_allow_logit_argpartition_router_topk()


@pytest.mark.cpu
def test_qwen_router_argpartition_logits_topk_opt_in(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZMLX_QWEN_ROUTER_ARGPARTITION_LOGITS_TOPK", "1")
    assert moe_mlp._qwen_allow_logit_argpartition_router_topk()


@pytest.mark.cpu
def test_qwen_combine_mode_default_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZMLX_QWEN_COMBINE_MODE", raising=False)
    assert moe_mlp._qwen_combine_mode() == "off"


@pytest.mark.cpu
def test_qwen_combine_mode_valid(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZMLX_QWEN_COMBINE_MODE", "fp32_no_fma")
    assert moe_mlp._qwen_combine_mode() == "fp32_no_fma"


@pytest.mark.cpu
def test_qwen_combine_mode_invalid_falls_back(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZMLX_QWEN_COMBINE_MODE", "weird")
    assert moe_mlp._qwen_combine_mode() == "off"


@pytest.mark.cpu
def test_glm_combine_mode_default_fp32_no_fma(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZMLX_GLM_COMBINE_MODE", raising=False)
    assert moe_mlp._glm_combine_mode() == "fp32_no_fma"


@pytest.mark.cpu
def test_glm_combine_mode_valid(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZMLX_GLM_COMBINE_MODE", "exact")
    assert moe_mlp._glm_combine_mode() == "exact"


@pytest.mark.cpu
def test_glm_combine_mode_invalid_falls_back(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZMLX_GLM_COMBINE_MODE", "weird")
    assert moe_mlp._glm_combine_mode() == "fp32_no_fma"


class _FakeGateUpProj:
    def __init__(self, d_hidden: int):
        self.d_hidden = int(d_hidden)

    def __call__(self, x, indices):
        b = int(indices.shape[0])
        k = int(indices.shape[1])
        base = x[:, : self.d_hidden].astype(mx.float16)
        return mx.broadcast_to(base[:, None, :], (b, k, self.d_hidden))


class _FakeGateUpProjRank5:
    def __init__(self, d_hidden: int):
        self.d_hidden = int(d_hidden)

    def __call__(self, x, indices):
        # Mimic Qwen/SwitchGLU projection rank: (..., K, 1, D_hidden)
        out_shape = tuple(indices.shape) + (1, self.d_hidden)
        base = x[..., : self.d_hidden].astype(mx.float16)
        base = mx.expand_dims(base, axis=-2)  # (..., 1, D_hidden)
        base = mx.expand_dims(base, axis=-2)  # (..., 1, 1, D_hidden)
        return mx.broadcast_to(base, out_shape)


class _FakeDownProj:
    def __init__(self, n_experts: int, d_hidden: int, d_out: int):
        self.weight = mx.random.normal((n_experts, d_hidden, d_out)).astype(mx.float16)
        self.bias = None


class _FakeSwitchMLP:
    def __init__(self, n_experts: int = 8, d_hidden: int = 8, d_out: int = 6):
        self.gate_proj = _FakeGateUpProj(d_hidden)
        self.up_proj = _FakeGateUpProj(d_hidden)
        self.down_proj = _FakeDownProj(n_experts, d_hidden, d_out)
        self.activation = None


class _FakeSwitchMLPRank5(_FakeSwitchMLP):
    def __init__(self, n_experts: int = 8, d_hidden: int = 8, d_out: int = 6):
        self.gate_proj = _FakeGateUpProjRank5(d_hidden)
        self.up_proj = _FakeGateUpProjRank5(d_hidden)
        self.down_proj = _FakeDownProj(n_experts, d_hidden, d_out)
        self.activation = None


@pytest.mark.cpu
def test_flatten_for_fused_combine_accepts_singleton_axis():
    act = mx.random.normal((2, 3, 4, 1, 6)).astype(mx.float16)
    gate = mx.softmax(mx.random.normal((2, 3, 4)), axis=-1).astype(mx.float32)
    indices = mx.zeros((2, 3, 4), dtype=mx.uint32)

    flat = moe_mlp._flatten_for_fused_combine(act, gate, indices)
    assert flat is not None
    act_flat, gate_flat, idx_flat, out_shape = flat
    assert tuple(act_flat.shape) == (6, 4, 6)
    assert tuple(gate_flat.shape) == (6, 4)
    assert tuple(idx_flat.shape) == (6, 4)
    assert tuple(out_shape) == (2, 3)


@pytest.mark.cpu
def test_try_fused_downproj_combine_fp32_gate_falls_back(monkeypatch: pytest.MonkeyPatch):
    switch_mlp = _FakeSwitchMLP()
    x = mx.random.normal((2, 8)).astype(mx.float16)
    indices = mx.array([[0, 1], [2, 3]], dtype=mx.uint32)
    gate = mx.ones((2, 2), dtype=mx.float16)

    monkeypatch.setattr(moe_mlp, "_disc_swiglu", None)
    monkeypatch.setattr(moe_mlp.transformer, "swiglu2", lambda g, u: g * u)

    called = {"value": False, "kwargs": {}}

    def _fake_gather(*args, **kwargs):
        called["value"] = True
        called["kwargs"] = kwargs
        return mx.zeros((2, 6), dtype=mx.float16)

    monkeypatch.setattr(moe_mlp.moe, "gather_qmm_combine", _fake_gather)

    out = moe_mlp._try_fused_downproj_combine(
        switch_mlp,
        x,
        indices,
        gate,
        require_fp32=True,
    )
    assert out is not None
    assert called["value"] is True
    kwargs = called["kwargs"]
    assert kwargs.get("output_dtype") is None
    assert kwargs.get("no_fma") is False


@pytest.mark.cpu
def test_try_fused_downproj_combine_fp32_route(monkeypatch: pytest.MonkeyPatch):
    switch_mlp = _FakeSwitchMLP()
    x = mx.random.normal((2, 8)).astype(mx.float16)
    indices = mx.array([[0, 1], [2, 3]], dtype=mx.uint32)
    gate = mx.softmax(mx.random.normal((2, 2)), axis=-1).astype(mx.float32)

    monkeypatch.setattr(moe_mlp, "_disc_swiglu", None)
    monkeypatch.setattr(moe_mlp.transformer, "swiglu2", lambda g, u: g * u)

    called: dict[str, object] = {}

    def _fake_gather(act, weights, gate_weights, inds, **kwargs):
        called["kwargs"] = kwargs
        b = int(act.shape[0])
        d_out = int(weights.shape[2])
        out_dtype = kwargs.get("output_dtype", act.dtype)
        return mx.zeros((b, d_out), dtype=out_dtype)

    monkeypatch.setattr(moe_mlp.moe, "gather_qmm_combine", _fake_gather)

    out = moe_mlp._try_fused_downproj_combine(
        switch_mlp,
        x,
        indices,
        gate,
        require_fp32=True,
    )

    assert out is not None
    assert out.dtype == mx.float32
    kwargs = called.get("kwargs")
    assert isinstance(kwargs, dict)
    assert kwargs.get("output_dtype") == mx.float32
    assert kwargs.get("no_fma") is True


@pytest.mark.cpu
def test_try_fused_downproj_combine_forwards_vectorized_k(monkeypatch: pytest.MonkeyPatch):
    switch_mlp = _FakeSwitchMLP()
    x = mx.random.normal((2, 8)).astype(mx.float16)
    indices = mx.array([[0, 1], [2, 3]], dtype=mx.uint32)
    gate = mx.softmax(mx.random.normal((2, 2)), axis=-1).astype(mx.float32)

    monkeypatch.setattr(moe_mlp, "_disc_swiglu", None)
    monkeypatch.setattr(moe_mlp.transformer, "swiglu2", lambda g, u: g * u)

    called: dict[str, object] = {}

    def _fake_gather(act, weights, gate_weights, inds, **kwargs):
        called["kwargs"] = kwargs
        b = int(act.shape[0])
        d_out = int(weights.shape[2])
        out_dtype = kwargs.get("output_dtype", act.dtype)
        return mx.zeros((b, d_out), dtype=out_dtype)

    monkeypatch.setattr(moe_mlp.moe, "gather_qmm_combine", _fake_gather)

    out = moe_mlp._try_fused_downproj_combine(
        switch_mlp,
        x,
        indices,
        gate,
        require_fp32=True,
        vectorized_k=True,
    )

    assert out is not None
    kwargs = called.get("kwargs")
    assert isinstance(kwargs, dict)
    assert kwargs.get("vectorized_k") is True


@pytest.mark.cpu
def test_try_fused_downproj_combine_rank5_activation(monkeypatch: pytest.MonkeyPatch):
    switch_mlp = _FakeSwitchMLPRank5()
    x = mx.random.normal((1, 1, 8)).astype(mx.float16)
    indices = mx.array([[[0, 1]]], dtype=mx.uint32)
    gate = mx.softmax(mx.random.normal((1, 1, 2)), axis=-1).astype(mx.float32)

    monkeypatch.setattr(moe_mlp, "_disc_swiglu", None)
    monkeypatch.setattr(moe_mlp.transformer, "swiglu2", lambda g, u: g * u)

    called: dict[str, object] = {}

    def _fake_gather(act, weights, gate_weights, inds, **kwargs):
        called["act_shape"] = tuple(int(v) for v in act.shape)
        b = int(act.shape[0])
        d_out = int(weights.shape[2])
        out_dtype = kwargs.get("output_dtype", act.dtype)
        return mx.zeros((b, d_out), dtype=out_dtype)

    monkeypatch.setattr(moe_mlp.moe, "gather_qmm_combine", _fake_gather)

    out = moe_mlp._try_fused_downproj_combine(
        switch_mlp,
        x,
        indices,
        gate,
        require_fp32=True,
    )

    assert out is not None
    assert tuple(out.shape) == (1, 1, 6)
    assert called.get("act_shape") == (1, 2, 8)


class _FakeQwenGateModule:
    norm_topk_prob = True

    def gate(self, x):
        return x


@pytest.mark.cpu
def test_qwen_gating_fused_router_opt_in(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZMLX_QWEN_FUSED_ROUTER_TOPK", "1")

    called: dict[str, object] = {}

    def _fake_topk(logits, **kwargs):
        called["kwargs"] = kwargs
        b = int(logits.shape[0])
        k = int(kwargs["k"])
        w = mx.full((b, k), 1.0 / k, dtype=mx.float32)
        idx = mx.zeros((b, k), dtype=mx.uint32)
        return w, idx

    monkeypatch.setattr(moe_mlp.moe, "topk_gating_softmax", _fake_topk)

    mod = _FakeQwenGateModule()
    x = mx.random.normal((2, 8)).astype(mx.float16)
    idx, weights = moe_mlp._gating(mod, x, "gate", 2, is_qwen3=True)

    assert idx.dtype == mx.uint32
    assert weights.dtype == mx.float32
    kwargs = called.get("kwargs")
    assert isinstance(kwargs, dict)
    assert kwargs.get("norm_topk_prob") is True


@pytest.mark.cpu
def test_qwen_gating_logit_argpartition_opt_in(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZMLX_QWEN_FUSED_ROUTER_TOPK", raising=False)
    monkeypatch.setenv("ZMLX_QWEN_ROUTER_ARGPARTITION_LOGITS", "1")

    mod = _FakeQwenGateModule()
    x = mx.array(
        [
            [0.5, 1.0, -0.2, 0.1],
            [3.0, 1.0, 2.0, -4.0],
        ],
        dtype=mx.float32,
    )

    idx, weights = moe_mlp._gating(mod, x, "gate", 2, is_qwen3=True)

    expected_idx = mx.argpartition(x, kth=-2, axis=-1)[..., -2:].astype(mx.uint32)
    expected_weights = mx.softmax(
        mx.take_along_axis(x, expected_idx, axis=-1),
        axis=-1,
        precise=True,
    )

    assert bool((idx == expected_idx).all().item())
    assert bool(mx.allclose(weights, expected_weights, rtol=0.0, atol=0.0).item())


@pytest.mark.cpu
def test_qwen_gating_logit_argpartition_topk_opt_in(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZMLX_QWEN_FUSED_ROUTER_TOPK", raising=False)
    monkeypatch.setenv("ZMLX_QWEN_ROUTER_ARGPARTITION_LOGITS", "1")
    monkeypatch.setenv("ZMLX_QWEN_ROUTER_ARGPARTITION_LOGITS_TOPK", "1")

    called: dict[str, object] = {}

    def _fake_router(logits, **kwargs):
        called["kwargs"] = kwargs
        b = int(logits.shape[0])
        k = int(kwargs["k"])
        weights = mx.full((b, k), 1.0 / k, dtype=mx.float32)
        idx = mx.zeros((b, k), dtype=mx.uint32)
        return weights, idx

    monkeypatch.setattr(moe_mlp.moe, "router_argpartition_logits_topk", _fake_router)

    mod = _FakeQwenGateModule()
    x = mx.random.normal((2, 8)).astype(mx.float32)
    idx, weights = moe_mlp._gating(mod, x, "gate", 2, is_qwen3=True)

    assert idx.dtype == mx.uint32
    assert weights.dtype == mx.float32
    kwargs = called.get("kwargs")
    assert isinstance(kwargs, dict)
    assert kwargs.get("k") == 2
