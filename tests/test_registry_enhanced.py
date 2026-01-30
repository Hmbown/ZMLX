"""Tests for the enhanced KernelRegistry."""

from __future__ import annotations

from zmlx.registry import GLOBAL_REGISTRY, KernelEntry, KernelRegistry


def test_registry_basic():
    reg = KernelRegistry()
    assert reg.size() == 0

    entry = KernelEntry(name="test_kernel", has_vjp=True, tags=frozenset({"norm"}))
    reg.register(entry)
    assert reg.size() == 1

    found = reg.get("test_kernel")
    assert found is not None
    assert found.name == "test_kernel"
    assert found.has_vjp is True


def test_registry_list_by_tag():
    reg = KernelRegistry()
    reg.register(KernelEntry(name="k1", tags=frozenset({"norm"})))
    reg.register(KernelEntry(name="k2", tags=frozenset({"activation"})))
    reg.register(KernelEntry(name="k3", tags=frozenset({"norm", "fused"})))

    norms = reg.list_kernels(tag="norm")
    assert len(norms) == 2
    assert {e.name for e in norms} == {"k1", "k3"}


def test_registry_by_pattern():
    reg = KernelRegistry()
    reg.register(KernelEntry(name="k1", pattern="elementwise_unary"))
    reg.register(KernelEntry(name="k2", pattern="rowwise_mapreduce"))
    reg.register(KernelEntry(name="k3", pattern="elementwise_unary"))

    unary = reg.by_pattern("elementwise_unary")
    assert len(unary) == 2


def test_registry_with_vjp():
    reg = KernelRegistry()
    reg.register(KernelEntry(name="k1", has_vjp=True))
    reg.register(KernelEntry(name="k2", has_vjp=False))
    reg.register(KernelEntry(name="k3", has_vjp=True))

    with_vjp = reg.with_vjp()
    assert len(with_vjp) == 2


def test_registry_hottest():
    reg = KernelRegistry()
    e1 = KernelEntry(name="k1", run_count=100)
    e2 = KernelEntry(name="k2", run_count=50)
    e3 = KernelEntry(name="k3", run_count=200)
    reg.register(e1)
    reg.register(e2)
    reg.register(e3)

    top = reg.hottest(2)
    assert len(top) == 2
    assert top[0].name == "k3"
    assert top[1].name == "k1"


def test_entry_touch():
    entry = KernelEntry(name="k1")
    assert entry.run_count == 0
    entry.touch()
    assert entry.run_count == 1
    assert entry.last_used > 0


def test_registry_clear():
    reg = KernelRegistry()
    reg.register(KernelEntry(name="k1"))
    reg.register(KernelEntry(name="k2"))
    assert reg.size() == 2
    reg.clear()
    assert reg.size() == 0


def test_registry_summary():
    reg = KernelRegistry()
    reg.register(KernelEntry(name="k1", pattern="elementwise_unary", has_vjp=True))
    reg.register(KernelEntry(name="k2", pattern="rowwise_mapreduce", has_vjp=True))
    reg.register(KernelEntry(name="k3", pattern="elementwise_unary"))

    summary = reg.summary()
    assert "3 kernels" in summary
    assert "2 with VJP" in summary
    assert "elementwise_unary: 2" in summary


def test_global_registry_exists():
    assert GLOBAL_REGISTRY is not None
    assert isinstance(GLOBAL_REGISTRY, KernelRegistry)


def test_legacy_api_still_works():
    """Ensure backward-compatible API is unchanged."""
    from zmlx.registry import cache_size, list_kernels

    # These should work without error
    names = list_kernels()
    assert isinstance(names, list)

    size = cache_size()
    assert isinstance(size, int)

    # clear_cache() should not raise
    # (but we don't actually call it to avoid clearing real cache)
