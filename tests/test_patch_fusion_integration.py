from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from zmlx.patch._traversal import apply_patterns
from zmlx.patch._types import FusionConfig, PatchConfig


class _ToyReduceModule(nn.Module):
    def __call__(self, x: Any, w: Any) -> Any:
        return mx.sum(x * w, axis=-1)


class _ToyUnsupportedAxisModule(nn.Module):
    def __call__(self, x: Any, w: Any) -> Any:
        return mx.sum(x * w, axis=0)


class _ToyModel(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def __call__(self, x: Any, w: Any) -> Any:
        return self.block(x, w)


class _ToyListModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = [_ToyReduceModule(), _ToyReduceModule()]

    def children(self) -> dict[str, Any]:
        return {"layers": self.layers}

    def __call__(self, x: Any, w: Any) -> Any:
        return self.layers[0](x, w) + self.layers[1](x, w)


class _ToyMixedListModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = [_ToyUnsupportedAxisModule(), _ToyReduceModule()]

    def children(self) -> dict[str, Any]:
        return {"layers": self.layers}

    def __call__(self, x: Any, w: Any) -> Any:
        return self.layers[0](x, w) + self.layers[1](x, w)


class _ToyPattern:
    @property
    def name(self) -> str:
        return "toy_pattern"

    def matches(self, module: Any, name: str, parent: Any | None = None) -> bool:
        return isinstance(module, (_ToyReduceModule, _ToyUnsupportedAxisModule))

    def apply(self, module: Any, config: PatchConfig) -> Any:
        original_call = module.__call__.__func__ if hasattr(module.__call__, "__func__") else None

        def patched_call(self_mod: Any, x: Any, w: Any) -> Any:
            if original_call is None:
                raise RuntimeError("toy pattern requires __call__.__func__")
            return original_call(self_mod, x, w)

        module._zmlx_original_call = original_call
        module.__class__ = type(
            module.__class__.__name__,
            (module.__class__,),
            {"__call__": patched_call},
        )
        return module


def test_patch_traversal_opt_in_fusion_validation_wraps_and_validates_real_call_path():
    model = _ToyModel(_ToyReduceModule())
    config = PatchConfig(
        fusion=FusionConfig(enabled=True, validate=True, patterns=("toy_pattern",)),
        verbose=False,
    )
    result = apply_patterns(model, [_ToyPattern()], config)
    assert result.patched_count == 1
    assert result.fusion_wrapped_count == 1

    x = mx.random.normal((5, 16)).astype(mx.float32)
    w = mx.random.normal((5, 1)).astype(mx.float32)
    out = model(x, w)
    out_block = model.block(x, w)
    expected = mx.sum(x * w, axis=-1)
    mx.eval(out, out_block, expected)
    assert mx.allclose(out, expected, rtol=1e-5, atol=1e-5).item()
    assert mx.allclose(out_block, expected, rtol=1e-5, atol=1e-5).item()
    assert len(config.fusion_state.validated) == 1
    assert len(config.fusion_state.blacklist) == 0


def test_patch_traversal_fallback_blacklists_unsupported_signature_real_call_path():
    model = _ToyModel(_ToyUnsupportedAxisModule())
    config = PatchConfig(
        fusion=FusionConfig(enabled=True, validate=True, patterns=("toy_pattern",)),
        verbose=False,
    )
    result = apply_patterns(model, [_ToyPattern()], config)
    assert result.patched_count == 1
    assert result.fusion_wrapped_count == 1

    x = mx.random.normal((4, 12)).astype(mx.float32)
    w = mx.random.normal((4, 12)).astype(mx.float32)

    out1 = model(x, w)
    expected1 = mx.sum(x * w, axis=0)
    mx.eval(out1, expected1)
    assert mx.allclose(out1, expected1, rtol=1e-5, atol=1e-5).item()
    assert config.fusion_state.compile_failures >= 1
    assert len(config.fusion_state.blacklist) == 1

    hits_before = config.fusion_state.blacklist_hits
    out2 = model(x, w)
    expected2 = mx.sum(x * w, axis=0)
    mx.eval(out2, expected2)
    assert mx.allclose(out2, expected2, rtol=1e-5, atol=1e-5).item()
    assert config.fusion_state.blacklist_hits >= hits_before + 1


def test_patch_traversal_list_modules_wraps_real_call_path():
    model = _ToyListModel()
    config = PatchConfig(
        fusion=FusionConfig(enabled=True, validate=True, patterns=("toy_pattern",)),
        verbose=False,
    )
    result = apply_patterns(model, [_ToyPattern()], config)
    assert result.patched_count == 2
    assert result.fusion_wrapped_count == 2

    x = mx.random.normal((3, 8)).astype(mx.float32)
    w = mx.random.normal((3, 1)).astype(mx.float32)
    out = model(x, w)
    expected = (mx.sum(x * w, axis=-1) * 2).astype(mx.float32)
    mx.eval(out, expected)
    assert mx.allclose(out, expected, rtol=1e-5, atol=1e-5).item()
    assert len(config.fusion_state.validated) == 2
    assert len(config.fusion_state.blacklist) == 0


def test_patch_traversal_blacklist_isolation_is_module_instance_aware():
    model = _ToyMixedListModel()
    config = PatchConfig(
        fusion=FusionConfig(enabled=True, validate=True, patterns=("toy_pattern",)),
        verbose=False,
    )
    result = apply_patterns(model, [_ToyPattern()], config)
    assert result.patched_count == 2
    assert result.fusion_wrapped_count == 2

    x = mx.random.normal((4, 4)).astype(mx.float32)
    w = mx.random.normal((4, 1)).astype(mx.float32)

    out = model(x, w)
    expected = mx.sum(x * w, axis=0) + mx.sum(x * w, axis=-1)
    mx.eval(out, expected)
    assert mx.allclose(out, expected, rtol=1e-5, atol=1e-5).item()

    # Unsupported layer should blacklist only its own module scope.
    assert config.fusion_state.compile_failures >= 1
    assert len(config.fusion_state.blacklist) == 1
    assert len(config.fusion_state.validated) == 1


def test_patch_traversal_cross_instance_no_blacklist_poisoning():
    config1 = PatchConfig(
        fusion=FusionConfig(enabled=True, validate=True, patterns=("toy_pattern",)),
        verbose=False,
    )
    config2 = PatchConfig(
        fusion=FusionConfig(enabled=True, validate=True, patterns=("toy_pattern",)),
        verbose=False,
    )

    model1 = _ToyMixedListModel()
    result1 = apply_patterns(model1, [_ToyPattern()], config1)
    assert result1.patched_count == 2
    assert result1.fusion_wrapped_count == 2

    x = mx.random.normal((4, 4)).astype(mx.float32)
    w = mx.random.normal((4, 1)).astype(mx.float32)
    out1 = model1(x, w)
    expected1 = mx.sum(x * w, axis=0) + mx.sum(x * w, axis=-1)
    mx.eval(out1, expected1)
    assert mx.allclose(out1, expected1, rtol=1e-5, atol=1e-5).item()

    assert len(config1.fusion_state.blacklist) == 1
    assert len(config1.fusion_state.validated) == 1

    model2 = _ToyModel(_ToyReduceModule())
    result2 = apply_patterns(model2, [_ToyPattern()], config2)
    assert result2.patched_count == 1
    assert result2.fusion_wrapped_count == 1

    out2 = model2(x, w)
    expected2 = mx.sum(x * w, axis=-1)
    mx.eval(out2, expected2)
    assert mx.allclose(out2, expected2, rtol=1e-5, atol=1e-5).item()

    assert len(config2.fusion_state.blacklist) == 0
    assert len(config2.fusion_state.validated) == 1
    assert config2.fusion_state.compile_failures == 0


def test_patch_traversal_legacy_key_format_still_works():
    model = _ToyModel(_ToyReduceModule())
    config = PatchConfig(
        fusion=FusionConfig(enabled=True, validate=True, patterns=("toy_pattern",)),
        verbose=False,
    )

    result = apply_patterns(model, [_ToyPattern()], config)
    assert result.patched_count == 1
    assert result.fusion_wrapped_count == 1

    x = mx.random.normal((3, 8)).astype(mx.float32)
    w = mx.random.normal((3, 1)).astype(mx.float32)

    legacy_validated_key = (
        "toy_pattern",
        (("tensor", (3, 8), "mlx.core.float32"), ("tensor", (3, 1), "mlx.core.float32")),
    )
    config.fusion_state.validated.add(legacy_validated_key)

    out = model(x, w)
    expected = mx.sum(x * w, axis=-1)
    mx.eval(out, expected)
    assert mx.allclose(out, expected, rtol=1e-5, atol=1e-5).item()

    assert legacy_validated_key in config.fusion_state.validated

    model2 = _ToyModel(_ToyUnsupportedAxisModule())
    config2 = PatchConfig(
        fusion=FusionConfig(enabled=True, validate=True, patterns=("toy_pattern",)),
        verbose=False,
    )
    result2 = apply_patterns(model2, [_ToyPattern()], config2)
    assert result2.patched_count == 1
    assert result2.fusion_wrapped_count == 1

    legacy_blacklist_key = (
        "toy_pattern",
        (("tensor", (3, 8), "mlx.core.float32"), ("tensor", (3, 1), "mlx.core.float32")),
    )
    config2.fusion_state.blacklist.add(legacy_blacklist_key)

    out2 = model2(x, w)
    expected2 = mx.sum(x * w, axis=0)
    mx.eval(out2, expected2)
    assert mx.allclose(out2, expected2, rtol=1e-5, atol=1e-5).item()

    assert config2.fusion_state.blacklist_hits >= 1


def test_patch_traversal_deterministic_module_instance_key():
    model1 = _ToyModel(_ToyReduceModule())
    model2 = _ToyModel(_ToyReduceModule())

    from zmlx.patch._traversal import _module_fingerprint

    fp1 = _module_fingerprint(model1.block)
    fp2 = _module_fingerprint(model2.block)

    assert fp1 == fp2
    assert "_ToyReduceModule" in fp1
    assert ":p" in fp1
    assert ":c" in fp1


def test_patch_traversal_different_structure_different_key():
    model1 = _ToyModel(_ToyReduceModule())
    model2 = _ToyModel(_ToyUnsupportedAxisModule())

    from zmlx.patch._traversal import _module_fingerprint

    fp1 = _module_fingerprint(model1.block)
    fp2 = _module_fingerprint(model2.block)

    assert fp1 != fp2


def test_patch_traversal_legacy_key_normalization_opt_in():
    model = _ToyModel(_ToyReduceModule())
    config = PatchConfig(
        fusion=FusionConfig(
            enabled=True,
            validate=True,
            patterns=("toy_pattern",),
            normalize_legacy_keys=True,
        ),
        verbose=False,
    )

    result = apply_patterns(model, [_ToyPattern()], config)
    assert result.patched_count == 1
    assert result.fusion_wrapped_count == 1

    x = mx.random.normal((3, 8)).astype(mx.float32)
    w = mx.random.normal((3, 1)).astype(mx.float32)

    legacy_key = (
        "toy_pattern",
        (("tensor", (3, 8), "mlx.core.float32"), ("tensor", (3, 1), "mlx.core.float32")),
    )
    config.fusion_state.validated.add(legacy_key)
    assert legacy_key in config.fusion_state.validated

    out = model(x, w)
    expected = mx.sum(x * w, axis=-1)
    mx.eval(out, expected)
    assert mx.allclose(out, expected, rtol=1e-5, atol=1e-5).item()

    assert legacy_key not in config.fusion_state.validated
    assert len(config.fusion_state.validated) == 1
    migrated_key = next(iter(config.fusion_state.validated))
    assert len(migrated_key) == 3
    assert migrated_key[0] == "toy_pattern"
    assert migrated_key[1].startswith("block#")


def test_patch_traversal_legacy_key_no_normalization_by_default():
    model = _ToyModel(_ToyReduceModule())
    config = PatchConfig(
        fusion=FusionConfig(
            enabled=True,
            validate=True,
            patterns=("toy_pattern",),
            normalize_legacy_keys=False,
        ),
        verbose=False,
    )

    result = apply_patterns(model, [_ToyPattern()], config)
    assert result.patched_count == 1
    assert result.fusion_wrapped_count == 1

    x = mx.random.normal((3, 8)).astype(mx.float32)
    w = mx.random.normal((3, 1)).astype(mx.float32)

    legacy_key = (
        "toy_pattern",
        (("tensor", (3, 8), "mlx.core.float32"), ("tensor", (3, 1), "mlx.core.float32")),
    )
    config.fusion_state.validated.add(legacy_key)
    assert legacy_key in config.fusion_state.validated

    out = model(x, w)
    expected = mx.sum(x * w, axis=-1)
    mx.eval(out, expected)
    assert mx.allclose(out, expected, rtol=1e-5, atol=1e-5).item()

    assert legacy_key in config.fusion_state.validated


def test_patch_traversal_blacklist_normalization_opt_in():
    model = _ToyModel(_ToyUnsupportedAxisModule())
    config = PatchConfig(
        fusion=FusionConfig(
            enabled=True,
            validate=True,
            patterns=("toy_pattern",),
            normalize_legacy_keys=True,
        ),
        verbose=False,
    )

    result = apply_patterns(model, [_ToyPattern()], config)
    assert result.patched_count == 1
    assert result.fusion_wrapped_count == 1

    x = mx.random.normal((3, 8)).astype(mx.float32)
    w = mx.random.normal((3, 1)).astype(mx.float32)

    legacy_blacklist_key = (
        "toy_pattern",
        (("tensor", (3, 8), "mlx.core.float32"), ("tensor", (3, 1), "mlx.core.float32")),
    )
    config.fusion_state.blacklist.add(legacy_blacklist_key)
    assert legacy_blacklist_key in config.fusion_state.blacklist

    out = model(x, w)
    expected = mx.sum(x * w, axis=0)
    mx.eval(out, expected)
    assert mx.allclose(out, expected, rtol=1e-5, atol=1e-5).item()

    assert config.fusion_state.blacklist_hits >= 1
    assert legacy_blacklist_key not in config.fusion_state.blacklist
    migrated_key = next(iter(config.fusion_state.blacklist))
    assert len(migrated_key) == 3
    assert migrated_key[0] == "toy_pattern"
    assert migrated_key[1].startswith("block#")
