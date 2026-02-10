from zmlx.patch.patterns.deepseek_router import _is_deepseek_gate_module


def _make_gate(module_path: str):
    gate_cls = type("MoEGate", (), {})
    gate_cls.__module__ = module_path
    return gate_cls()


def test_deepseek_gate_module_match_includes_kimi() -> None:
    assert _is_deepseek_gate_module(_make_gate("mlx_lm.models.deepseek_v3"))
    assert _is_deepseek_gate_module(_make_gate("mlx_lm.models.deepseek_v32"))
    assert _is_deepseek_gate_module(_make_gate("mlx_lm.models.kimi_k25"))


def test_deepseek_gate_module_match_rejects_non_moe_gate() -> None:
    cls = type("Router", (), {})
    cls.__module__ = "mlx_lm.models.kimi_k25"
    assert not _is_deepseek_gate_module(cls())
