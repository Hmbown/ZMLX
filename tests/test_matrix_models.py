from zmlx.matrix.models import _infer_quant


def test_infer_quant_handles_decimal_bit_suffixes() -> None:
    assert _infer_quant("mlx-community/Kimi-K2.5-MLX-3.6bit") == "3.6bit"
    assert _infer_quant("mlx-community/Kimi-K2.5-MLX-4.2bit") == "4.2bit"


def test_infer_quant_preserves_integer_bit_suffixes() -> None:
    assert _infer_quant("mlx-community/Kimi-K2.5") == "4bit"
    assert _infer_quant("mlx-community/Qwen3-30B-A3B-4bit") == "4bit"
