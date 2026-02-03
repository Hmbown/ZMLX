import pytest

from zmlx.patch.patterns import moe_mlp


@pytest.mark.cpu
def test_moe_fused_swiglu_token_gate():
    assert moe_mlp._should_fuse_swiglu_tokens(1, 1)
    assert not moe_mlp._should_fuse_swiglu_tokens(2, 1)
