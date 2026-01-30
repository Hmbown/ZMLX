"""Tests for persistent autotune cache."""

from __future__ import annotations

import json

from zmlx.autotune import (
    GLOBAL_AUTOTUNE_CACHE,
    AutotuneConfig,
    AutotuneKey,
    load_autotune_cache,
    save_autotune_cache,
)


def test_save_and_load(tmp_path):
    """Test roundtrip: save -> clear -> load."""
    path = str(tmp_path / "autotune_test.json")

    # Insert some entries
    key1 = AutotuneKey(
        kernel_name="kk_softmax_D128_TG256",
        input_shapes=((4, 128),),
        input_dtypes=("float32",),
        grid=(1024, 1, 1),
    )
    key2 = AutotuneKey(
        kernel_name="kk_rmsnorm_D64_TG128",
        input_shapes=((8, 64),),
        input_dtypes=("float16",),
        grid=(1024, 1, 1),
    )

    GLOBAL_AUTOTUNE_CACHE[key1] = AutotuneConfig(threadgroup=(256, 1, 1))
    GLOBAL_AUTOTUNE_CACHE[key2] = AutotuneConfig(threadgroup=(128, 1, 1))

    # Save
    save_autotune_cache(path)

    # Verify file exists and is valid JSON
    with open(path) as f:
        data = json.load(f)
    assert len(data) > 0

    # Clear and reload
    del GLOBAL_AUTOTUNE_CACHE[key1]
    del GLOBAL_AUTOTUNE_CACHE[key2]

    count = load_autotune_cache(path)
    assert count >= 2

    assert key1 in GLOBAL_AUTOTUNE_CACHE
    assert GLOBAL_AUTOTUNE_CACHE[key1].threadgroup == (256, 1, 1)
    assert key2 in GLOBAL_AUTOTUNE_CACHE
    assert GLOBAL_AUTOTUNE_CACHE[key2].threadgroup == (128, 1, 1)

    # Cleanup
    del GLOBAL_AUTOTUNE_CACHE[key1]
    del GLOBAL_AUTOTUNE_CACHE[key2]


def test_load_nonexistent(tmp_path):
    """Loading from a non-existent file returns 0."""
    count = load_autotune_cache(str(tmp_path / "nonexistent.json"))
    assert count == 0


def test_load_corrupt_file(tmp_path):
    """Loading from a corrupt file returns 0."""
    path = str(tmp_path / "corrupt.json")
    with open(path, "w") as f:
        f.write("not valid json {{{")

    count = load_autotune_cache(path)
    assert count == 0


def test_device_cache_key():
    """Device cache key includes family and MLX version."""
    from zmlx.autotune import _device_cache_key

    key = _device_cache_key()
    assert isinstance(key, str)
    assert len(key) > 0
