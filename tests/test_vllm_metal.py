"""Tests for zmlx.vllm_metal integration module."""

from __future__ import annotations

import os
import sys
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _reset_module():
    """Reset the vllm_metal integration state between tests."""
    # Remove any cached imports
    for name in list(sys.modules):
        if "zmlx.vllm_metal" in name:
            sys.modules.pop(name, None)
    yield
    # Cleanup after test
    for name in list(sys.modules):
        if "zmlx.vllm_metal" in name:
            sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# Fixtures: fake vllm_metal model runners
# ---------------------------------------------------------------------------


class _FakeModel:
    """Minimal fake MLX model for testing."""

    def __init__(self):
        self.patched = False


class _FakeV1ModelRunner:
    """Fake vllm_metal.v1.model_runner.MetalModelRunner."""

    def __init__(self):
        self.model = None
        self.model_config = mock.MagicMock()
        self.model_config.model = "test-model/fake-7b"

    def load_model(self) -> None:
        self.model = _FakeModel()


class _FakeLegacyModelRunner:
    """Fake vllm_metal.model_runner.MetalModelRunner."""

    def __init__(self):
        self.model = None
        self.vllm_config = mock.MagicMock()
        self.vllm_config.model_config.model = "test-model/fake-7b"

    def load_model(self) -> None:
        self.model = _FakeModel()


_ORIGINAL_V1_LOAD_MODEL = _FakeV1ModelRunner.load_model
_ORIGINAL_LEGACY_LOAD_MODEL = _FakeLegacyModelRunner.load_model


def _install_fake_vllm_metal():
    """Install fake vllm_metal modules into sys.modules."""
    # Create fake module hierarchy
    vllm_metal_mod = mock.MagicMock()
    vllm_metal_v1_mod = mock.MagicMock()
    vllm_metal_v1_mr_mod = mock.MagicMock()
    vllm_metal_mr_mod = mock.MagicMock()

    # Wire up the model runner classes
    vllm_metal_v1_mr_mod.MetalModelRunner = _FakeV1ModelRunner
    vllm_metal_mr_mod.MetalModelRunner = _FakeLegacyModelRunner

    # Install into sys.modules
    sys.modules["vllm_metal"] = vllm_metal_mod
    sys.modules["vllm_metal.v1"] = vllm_metal_v1_mod
    sys.modules["vllm_metal.v1.model_runner"] = vllm_metal_v1_mr_mod
    sys.modules["vllm_metal.model_runner"] = vllm_metal_mr_mod

    # Make attribute access work
    vllm_metal_mod.v1 = vllm_metal_v1_mod
    vllm_metal_mod.model_runner = vllm_metal_mr_mod
    vllm_metal_v1_mod.model_runner = vllm_metal_v1_mr_mod


def _uninstall_fake_vllm_metal():
    """Remove fake vllm_metal modules from sys.modules."""
    for name in list(sys.modules):
        if name == "vllm_metal" or name.startswith("vllm_metal."):
            sys.modules.pop(name, None)
    _FakeV1ModelRunner.load_model = _ORIGINAL_V1_LOAD_MODEL
    _FakeLegacyModelRunner.load_model = _ORIGINAL_LEGACY_LOAD_MODEL


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEnable:
    """Test the enable() function."""

    def test_enable_wraps_v1_model_runner(self):
        _install_fake_vllm_metal()
        try:
            from zmlx.vllm_metal import enable

            enable()

            assert getattr(_FakeV1ModelRunner.load_model, "_zmlx_wrapped", False)
        finally:
            _uninstall_fake_vllm_metal()

    def test_enable_raises_without_vllm_metal(self):
        # Ensure vllm_metal is not available
        _uninstall_fake_vllm_metal()
        from zmlx.vllm_metal import enable

        with pytest.raises(ImportError, match="vllm-metal"):
            enable()

    def test_enable_is_idempotent(self):
        _install_fake_vllm_metal()
        try:
            from zmlx.vllm_metal import enable

            enable()
            enable()  # Should not raise
        finally:
            _uninstall_fake_vllm_metal()


class TestPatchModel:
    """Test that _patch_model correctly applies ZMLX patches."""

    def test_patch_model_calls_zmlx_patch(self):
        """Verify that _patch_model calls zmlx.patch.patch."""
        from zmlx.vllm_metal import _patch_model

        model = _FakeModel()

        # Mock zmlx.patch.patch
        with mock.patch("zmlx.patch.patch") as mock_patch:
            mock_patch.return_value = model
            result = _patch_model(model, "test-model")
            mock_patch.assert_called_once()
            assert result is model

    def test_patch_model_logs_patched_count(self):
        """The info log should use PatchResult.patched_count."""
        from zmlx.vllm_metal import _patch_model

        model = _FakeModel()
        model._zmlx_patch_result = mock.MagicMock(patched_count=7)

        with mock.patch("zmlx.patch.patch", return_value=model):
            with mock.patch("zmlx.vllm_metal.logger") as mock_logger:
                _patch_model(model, "test-model")
                assert mock_logger.info.call_args is not None
                assert mock_logger.info.call_args[0][2] == 7

    def test_patch_model_respects_patterns_env(self):
        """ZMLX_VLLM_PATTERNS should be forwarded."""
        from zmlx.vllm_metal import _patch_model

        model = _FakeModel()

        with mock.patch.dict(os.environ, {"ZMLX_VLLM_PATTERNS": "moe_mlp,swiglu_mlp"}):
            with mock.patch("zmlx.patch.patch") as mock_patch:
                mock_patch.return_value = model
                _patch_model(model, "test")
                call_kwargs = mock_patch.call_args[1]
                assert call_kwargs["patterns"] == ["moe_mlp", "swiglu_mlp"]

    def test_patch_model_respects_exclude_env(self):
        """ZMLX_VLLM_EXCLUDE should be forwarded."""
        from zmlx.vllm_metal import _patch_model

        model = _FakeModel()

        with mock.patch.dict(os.environ, {"ZMLX_VLLM_EXCLUDE": "rmsnorm"}):
            with mock.patch("zmlx.patch.patch") as mock_patch:
                mock_patch.return_value = model
                _patch_model(model, "test")
                call_kwargs = mock_patch.call_args[1]
                assert call_kwargs["exclude"] == ["rmsnorm"]

    def test_patch_model_survives_import_error(self):
        """If zmlx.patch is not available, should log and return model."""
        from zmlx.vllm_metal import _patch_model

        model = _FakeModel()

        with mock.patch.dict(sys.modules, {"zmlx.patch": None}):
            # This should not raise
            with mock.patch("zmlx.vllm_metal.logger"):
                result = _patch_model(model, "test")
                assert result is model


class TestRegister:
    """Test the vLLM plugin register() entry point."""

    def test_register_skips_without_env_var(self):
        """register() should be a no-op when ZMLX_VLLM is not set."""
        env = {k: v for k, v in os.environ.items() if k != "ZMLX_VLLM"}
        with mock.patch.dict(os.environ, env, clear=True):
            from zmlx.vllm_metal import register

            # Should not raise, should not enable
            register()

    def test_register_enables_with_env_var(self):
        _install_fake_vllm_metal()
        try:
            with mock.patch.dict(os.environ, {"ZMLX_VLLM": "1"}):
                from zmlx.vllm_metal import register

                register()
                # Should have enabled
                import zmlx.vllm_metal as mod

                assert mod._enabled
        finally:
            _uninstall_fake_vllm_metal()


class TestEndToEnd:
    """End-to-end test: enable + model load triggers ZMLX patch."""

    def test_v1_model_load_triggers_patch(self):
        _install_fake_vllm_metal()
        try:
            from zmlx.vllm_metal import enable

            enable()

            # Now create a runner and load a model
            runner = _FakeV1ModelRunner()

            with mock.patch("zmlx.patch.patch") as mock_patch:
                mock_patch.side_effect = lambda m, **kw: m
                runner.load_model()
                # zmlx.patch.patch should have been called
                mock_patch.assert_called_once()
                # model should still be set
                assert runner.model is not None
        finally:
            _uninstall_fake_vllm_metal()

    def test_legacy_model_load_triggers_patch(self):
        _install_fake_vllm_metal()
        try:
            from zmlx.vllm_metal import enable

            enable()

            runner = _FakeLegacyModelRunner()

            with mock.patch("zmlx.patch.patch") as mock_patch:
                mock_patch.side_effect = lambda m, **kw: m
                runner.load_model()
                mock_patch.assert_called_once()
                assert runner.model is not None
        finally:
            _uninstall_fake_vllm_metal()

    def test_disable_restores_original_methods(self):
        _install_fake_vllm_metal()
        try:
            from zmlx.vllm_metal import disable, enable

            enable()
            assert getattr(_FakeV1ModelRunner.load_model, "_zmlx_wrapped", False)
            disable()
            assert _FakeV1ModelRunner.load_model is _ORIGINAL_V1_LOAD_MODEL
            assert _FakeLegacyModelRunner.load_model is _ORIGINAL_LEGACY_LOAD_MODEL

            runner = _FakeV1ModelRunner()
            with mock.patch("zmlx.patch.patch") as mock_patch:
                runner.load_model()
                mock_patch.assert_not_called()
        finally:
            _uninstall_fake_vllm_metal()
