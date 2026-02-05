"""Tests for KernelFamily variant generators."""

from __future__ import annotations

from zmlx.dsl.family import Axis, KernelFamily


class TestAxis:
    def test_frozen(self):
        a = Axis("fma", (True, False))
        assert a.name == "fma"
        assert a.values == (True, False)


class TestKernelFamily:
    def test_variant_count(self):
        """2 axes × 2 values each = 4 variants."""
        family = KernelFamily(
            name="test",
            axes=[
                Axis("a", (True, False)),
                Axis("b", ("x", "y")),
            ],
            factory=lambda a, b: lambda: f"{a}_{b}",
        )
        assert len(family) == 4

    def test_variant_names_default(self):
        """Default namer uses name + axis values."""
        family = KernelFamily(
            name="kern",
            axes=[
                Axis("mode", ("fast", "safe")),
            ],
            factory=lambda mode: lambda: mode,
        )
        variants = family.variants()
        assert "kern_fast" in variants
        assert "kern_safe" in variants

    def test_custom_namer(self):
        """Custom namer generates specific names."""
        family = KernelFamily(
            name="moe_combine",
            axes=[
                Axis("fma", (True, False)),
                Axis("acc_dtype", ("float", "T")),
            ],
            factory=lambda fma, acc_dtype: lambda: (fma, acc_dtype),
            namer=lambda fma, acc_dtype: (
                f"moe_combine"
                f"{'_no_fma' if not fma else ''}"
                f"{'_exact' if acc_dtype == 'T' else ''}"
            ),
        )
        variants = family.variants()
        assert "moe_combine" in variants  # fma=True, acc_dtype="float"
        assert "moe_combine_no_fma" in variants
        assert "moe_combine_exact" in variants
        assert "moe_combine_no_fma_exact" in variants

    def test_get_variant(self):
        """Get a specific variant by axis values."""
        family = KernelFamily(
            name="test",
            axes=[Axis("x", (1, 2, 3))],
            factory=lambda x: lambda: x * 10,
        )
        builder = family.get(x=2)
        assert builder() == 20

    def test_get_missing_raises(self):
        """Getting a nonexistent variant raises KeyError."""
        family = KernelFamily(
            name="test",
            axes=[Axis("x", (1, 2))],
            factory=lambda x: lambda: x,
            namer=lambda x: f"v{x}",
        )
        import pytest

        with pytest.raises(KeyError, match="v99"):
            family.get(x=99)

    def test_caching(self):
        """Variants dict is cached after first call."""
        call_count = 0

        def counting_factory(x):
            nonlocal call_count
            call_count += 1
            return lambda: x

        family = KernelFamily(
            name="test",
            axes=[Axis("x", (1, 2))],
            factory=counting_factory,
        )

        v1 = family.variants()
        v2 = family.variants()
        assert v1 is v2
        assert call_count == 2  # called once per variant, not again

    def test_iteration(self):
        """Family supports iteration over (name, builder) pairs."""
        family = KernelFamily(
            name="test",
            axes=[Axis("x", (1, 2))],
            factory=lambda x: lambda: x,
        )
        items = list(family)
        assert len(items) == 2
        names = [name for name, _ in items]
        assert "test_1" in names
        assert "test_2" in names

    def test_three_axes(self):
        """3 axes with 2, 3, 2 values = 12 variants."""
        family = KernelFamily(
            name="kern",
            axes=[
                Axis("a", (True, False)),
                Axis("b", ("x", "y", "z")),
                Axis("c", (1, 2)),
            ],
            factory=lambda a, b, c: lambda: (a, b, c),
        )
        assert len(family) == 12
