"""Tests for zmlx.train.config."""

from __future__ import annotations

import pytest


def test_default_config():
    from zmlx.train.config import TrainConfig

    config = TrainConfig()
    assert config.model == ""
    assert config.lora is False
    assert config.batch_size == 4
    assert config.iters == 1000
    assert config.patch is True


def test_config_validate_no_model():
    from zmlx.train.config import TrainConfig

    config = TrainConfig()
    with pytest.raises(ValueError, match="model is required"):
        config.validate()


def test_config_validate_no_dataset():
    from zmlx.train.config import TrainConfig

    config = TrainConfig(model="test-model")
    with pytest.raises(ValueError, match="dataset is required"):
        config.validate()


def test_config_validate_lora_and_dora():
    from zmlx.train.config import TrainConfig

    config = TrainConfig(model="test", dataset="test", lora=True, dora=True)
    with pytest.raises(ValueError, match="Cannot use both LoRA and DoRA"):
        config.validate()


def test_config_merge_cli():
    from zmlx.train.config import TrainConfig

    config = TrainConfig()
    config.merge_cli({"model": "my-model", "lora": True, "nonexistent": "ignored"})
    assert config.model == "my-model"
    assert config.lora is True


def test_config_yaml_roundtrip(tmp_path):
    """Test YAML loading."""
    yaml_content = """
model: "test-model"
dataset: "test-dataset"
lora: true
lora_rank: 16
iters: 500
batch_size: 2
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml_content)

    from zmlx.train.config import TrainConfig

    config = TrainConfig.from_yaml(str(config_file))
    assert config.model == "test-model"
    assert config.dataset == "test-dataset"
    assert config.lora is True
    assert config.lora_rank == 16
    assert config.iters == 500
    assert config.batch_size == 2


def test_config_valid():
    from zmlx.train.config import TrainConfig

    config = TrainConfig(model="test-model", dataset="test-dataset")
    config.validate()  # Should not raise
