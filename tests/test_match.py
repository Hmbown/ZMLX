"""Tests for declarative pattern matcher."""

from __future__ import annotations

from zmlx.dsl.match import (
    DeclarativePattern,
    attr_name_matches,
    class_name_is,
    custom,
    has_attr,
    is_instance,
    module_path_contains,
    parent_has_attr,
)

# ---------------------------------------------------------------------------
# Synthetic modules for testing
# ---------------------------------------------------------------------------

class FakeModule:
    """A minimal fake module for predicate testing."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Qwen3MoeSparseMoeBlock(FakeModule):
    pass


# Make it look like it came from mlx_lm
Qwen3MoeSparseMoeBlock.__module__ = "mlx_lm.models.qwen3_moe"


class GptOssBlock(FakeModule):
    pass


GptOssBlock.__module__ = "mlx_lm.models.gpt_oss.moe"


class DenseBlock(FakeModule):
    pass


DenseBlock.__module__ = "mlx_lm.models.llama"


# ---------------------------------------------------------------------------
# Individual predicate tests
# ---------------------------------------------------------------------------

class TestHasAttr:
    def test_present(self):
        mod = FakeModule(gate=True, experts=True)
        assert has_attr("gate")(mod, "layer.0")

    def test_absent(self):
        mod = FakeModule(weights=True)
        assert not has_attr("gate")(mod, "layer.0")


class TestIsInstance:
    def test_match(self):
        mod = Qwen3MoeSparseMoeBlock()
        assert is_instance(Qwen3MoeSparseMoeBlock)(mod, "moe")

    def test_no_match(self):
        mod = DenseBlock()
        assert not is_instance(Qwen3MoeSparseMoeBlock)(mod, "moe")


class TestModulePathContains:
    def test_match(self):
        mod = Qwen3MoeSparseMoeBlock()
        assert module_path_contains("qwen3_moe")(mod, "moe")

    def test_no_match(self):
        mod = DenseBlock()
        assert not module_path_contains("qwen3_moe")(mod, "moe")


class TestClassName:
    def test_match(self):
        mod = Qwen3MoeSparseMoeBlock()
        assert class_name_is("Qwen3MoeSparseMoeBlock")(mod, "moe")

    def test_no_match(self):
        mod = DenseBlock()
        assert not class_name_is("Qwen3MoeSparseMoeBlock")(mod, "moe")


class TestAttrNameMatches:
    def test_match(self):
        mod = FakeModule()
        assert attr_name_matches(r"layer\.\d+")(mod, "layer.0")
        assert attr_name_matches(r"layer\.\d+")(mod, "layer.42")

    def test_no_match(self):
        mod = FakeModule()
        assert not attr_name_matches(r"layer\.\d+")(mod, "block.0")


class TestParentHasAttr:
    def test_with_parent(self):
        parent = FakeModule(shared_experts=True)
        mod = FakeModule()
        assert parent_has_attr("shared_experts")(mod, "moe", parent)

    def test_no_parent(self):
        mod = FakeModule()
        assert not parent_has_attr("shared_experts")(mod, "moe", None)

    def test_parent_missing_attr(self):
        parent = FakeModule()
        mod = FakeModule()
        assert not parent_has_attr("shared_experts")(mod, "moe", parent)


class TestCustom:
    def test_custom_fn(self):
        pred = custom(lambda mod, name, parent: "moe" in name)
        assert pred(FakeModule(), "moe_block")
        assert not pred(FakeModule(), "dense_block")


# ---------------------------------------------------------------------------
# Combinator tests
# ---------------------------------------------------------------------------

class TestAnd:
    def test_both_true(self):
        pred = class_name_is("Qwen3MoeSparseMoeBlock") & module_path_contains("qwen3_moe")
        mod = Qwen3MoeSparseMoeBlock()
        assert pred(mod, "moe")

    def test_left_false(self):
        pred = class_name_is("SomethingElse") & module_path_contains("qwen3_moe")
        mod = Qwen3MoeSparseMoeBlock()
        assert not pred(mod, "moe")

    def test_right_false(self):
        pred = class_name_is("Qwen3MoeSparseMoeBlock") & module_path_contains("gpt_oss")
        mod = Qwen3MoeSparseMoeBlock()
        assert not pred(mod, "moe")


class TestOr:
    def test_either(self):
        pred = module_path_contains("qwen3_moe") | module_path_contains("gpt_oss")
        assert pred(Qwen3MoeSparseMoeBlock(), "moe")
        assert pred(GptOssBlock(), "moe")
        assert not pred(DenseBlock(), "moe")


class TestNot:
    def test_invert(self):
        pred = ~module_path_contains("llama")
        assert pred(Qwen3MoeSparseMoeBlock(), "moe")
        assert not pred(DenseBlock(), "dense")


class TestComplex:
    def test_real_world_qwen3(self):
        """Replicate the _is_qwen3_moe_block check."""
        qwen3_moe = (
            class_name_is("Qwen3MoeSparseMoeBlock")
            | module_path_contains("qwen3_moe")
        )
        assert qwen3_moe(Qwen3MoeSparseMoeBlock(), "moe")
        assert not qwen3_moe(GptOssBlock(), "moe")

    def test_real_world_gpt_oss(self):
        """Replicate _is_gpt_oss_block check."""
        gpt_oss = module_path_contains("gpt_oss")
        assert gpt_oss(GptOssBlock(), "moe")
        assert not gpt_oss(DenseBlock(), "dense")


# ---------------------------------------------------------------------------
# DeclarativePattern (PatchPattern protocol)
# ---------------------------------------------------------------------------

class TestDeclarativePattern:
    def test_name(self):
        pattern = DeclarativePattern(
            "moe_mlp",
            has_attr("gate"),
            lambda mod, config: mod,
        )
        assert pattern.name == "moe_mlp"

    def test_matches(self):
        pattern = DeclarativePattern(
            "moe_mlp",
            has_attr("gate") & has_attr("experts"),
            lambda mod, config: mod,
        )
        mod_with = FakeModule(gate=True, experts=True)
        mod_without = FakeModule(gate=True)
        assert pattern.matches(mod_with, "block")
        assert not pattern.matches(mod_without, "block")

    def test_apply(self):
        applied = []

        def my_apply(mod, config):
            applied.append(mod)
            return "replaced"

        pattern = DeclarativePattern(
            "test",
            has_attr("x"),
            my_apply,
        )
        mod = FakeModule(x=1)
        result = pattern.apply(mod, {})
        assert result == "replaced"
        assert applied == [mod]

    def test_satisfies_protocol(self):
        """DeclarativePattern should satisfy the PatchPattern protocol."""
        from zmlx.patch._types import PatchPattern

        pattern = DeclarativePattern(
            "test",
            has_attr("x"),
            lambda mod, config: mod,
        )
        assert isinstance(pattern, PatchPattern)
