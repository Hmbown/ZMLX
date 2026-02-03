import os

import pytest

from zmlx.kv_cache import kv_cache_kwargs


def _clear_env(*names: str) -> None:
    for name in names:
        os.environ.pop(name, None)


def test_kv_cache_kwargs_defaults():
    _clear_env("ZMLX_KV_BITS", "ZMLX_KV_GROUP_SIZE", "ZMLX_QUANTIZED_KV_START")
    assert kv_cache_kwargs() == {}


def test_kv_cache_kwargs_from_env():
    _clear_env("ZMLX_KV_GROUP_SIZE", "ZMLX_QUANTIZED_KV_START")
    os.environ["ZMLX_KV_BITS"] = "8"
    try:
        kwargs = kv_cache_kwargs()
    finally:
        _clear_env("ZMLX_KV_BITS")
    assert kwargs["kv_bits"] == 8
    assert kwargs["kv_group_size"] == 64
    assert kwargs["quantized_kv_start"] == 0


def test_kv_cache_kwargs_explicit():
    _clear_env("ZMLX_KV_BITS", "ZMLX_KV_GROUP_SIZE", "ZMLX_QUANTIZED_KV_START")
    kwargs = kv_cache_kwargs(kv_bits=4, kv_group_size=128, quantized_kv_start=256)
    assert kwargs == {
        "kv_bits": 4,
        "kv_group_size": 128,
        "quantized_kv_start": 256,
    }


def test_kv_cache_kwargs_invalid_env():
    _clear_env("ZMLX_KV_BITS", "ZMLX_KV_GROUP_SIZE", "ZMLX_QUANTIZED_KV_START")
    os.environ["ZMLX_KV_BITS"] = "nope"
    try:
        with pytest.raises(ValueError):
            kv_cache_kwargs()
    finally:
        _clear_env("ZMLX_KV_BITS")
