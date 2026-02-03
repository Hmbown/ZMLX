import pytest

from zmlx.patch.patterns import swiglu_mlp


@pytest.mark.cpu
def test_tg_progressive_allowlist_default(monkeypatch):
    monkeypatch.delenv(swiglu_mlp._QSWIGLU_TG_ALLOWLIST_ENV, raising=False)
    monkeypatch.delenv(swiglu_mlp._QSWIGLU_TG_MIN_EPS_ENV, raising=False)

    assert swiglu_mlp._tg_progressive_allowed(1, 2048, 7168, 10.0)
    assert not swiglu_mlp._tg_progressive_allowed(1, 2048, 7168, 1.0)
    assert not swiglu_mlp._tg_progressive_allowed(2, 2048, 7168, 10.0)
    assert not swiglu_mlp._tg_progressive_allowed(1, 1024, 2048, 10.0)


@pytest.mark.cpu
def test_tg_progressive_allowlist_override(monkeypatch):
    monkeypatch.setenv(swiglu_mlp._QSWIGLU_TG_ALLOWLIST_ENV, "*")
    monkeypatch.setenv(swiglu_mlp._QSWIGLU_TG_MIN_EPS_ENV, "0")
    assert swiglu_mlp._tg_progressive_allowed(1, 1024, 2048, 0.0)


@pytest.mark.cpu
def test_tg_progressive_family_denylist(monkeypatch):
    class Dummy:
        __module__ = "mlx_lm.models.lfm2"

    monkeypatch.setenv(swiglu_mlp._QSWIGLU_TG_DENY_FAMILY_ENV, "lfm")
    assert not swiglu_mlp._tg_family_allowed(Dummy())

    monkeypatch.setenv(swiglu_mlp._QSWIGLU_TG_DENY_FAMILY_ENV, "")
    assert swiglu_mlp._tg_family_allowed(Dummy())
