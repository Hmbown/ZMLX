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
def test_qwen_fused_swiglu_opt_in(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZMLX_QWEN_FUSED_SWIGLU", "1")
    assert moe_mlp._qwen_allow_fused_swiglu()
