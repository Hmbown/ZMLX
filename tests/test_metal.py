from __future__ import annotations

import numpy as np

from zmlx import metal

_ENV_DISABLE = "ZMLX_METAL_DISABLE_ROW_CONTIGUOUS_KERNELS"
_ENV_TELEMETRY = "ZMLX_METAL_CONTIGUITY_TELEMETRY"
_ENV_TELEMETRY_KERNELS = "ZMLX_METAL_CONTIGUITY_TELEMETRY_KERNELS"


class _FakeLaunch:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> list[np.ndarray]:
        self.calls.append(kwargs)
        out_shape = tuple(kwargs["output_shapes"][0])  # type: ignore[index]
        out_dtype = kwargs["output_dtypes"][0]  # type: ignore[index]
        return [np.zeros(out_shape, dtype=out_dtype)]


class _FakeFast:
    def __init__(self) -> None:
        self.compile_calls: list[dict[str, object]] = []
        self.launches: list[_FakeLaunch] = []

    def metal_kernel(self, **kwargs: object) -> _FakeLaunch:
        self.compile_calls.append(kwargs)
        launch = _FakeLaunch()
        self.launches.append(launch)
        return launch


class _FakeMX:
    def __init__(self) -> None:
        self.fast = _FakeFast()

    def eval(self, *args: object) -> None:
        del args

    def synchronize(self) -> None:
        return None


def _build_kernel(monkeypatch, *, name: str, ensure_row_contiguous: bool = True) -> tuple[metal.MetalKernel, _FakeMX]:
    fake = _FakeMX()
    monkeypatch.setattr(metal, "import_mx", lambda: fake)
    k = metal.kernel(
        name=name,
        input_names=["x"],
        output_names=["y"],
        source="kernel void stub() {}",
        ensure_row_contiguous=ensure_row_contiguous,
        cache=False,
    )
    return k, fake


def test_kernel_keeps_row_contiguous_default(monkeypatch) -> None:
    monkeypatch.delenv(_ENV_DISABLE, raising=False)
    monkeypatch.delenv(_ENV_TELEMETRY, raising=False)
    monkeypatch.delenv(_ENV_TELEMETRY_KERNELS, raising=False)

    k, fake = _build_kernel(monkeypatch, name="kk_default")

    assert fake.fast.compile_calls[-1]["ensure_row_contiguous"] is True
    assert k.spec.requested_ensure_row_contiguous is True
    assert k.spec.ensure_row_contiguous is True


def test_kernel_disable_row_contiguous_by_name(monkeypatch) -> None:
    monkeypatch.setenv(_ENV_DISABLE, "kk_probe_*")
    monkeypatch.delenv(_ENV_TELEMETRY, raising=False)
    monkeypatch.delenv(_ENV_TELEMETRY_KERNELS, raising=False)

    k, fake = _build_kernel(monkeypatch, name="kk_probe_rope")

    assert fake.fast.compile_calls[-1]["ensure_row_contiguous"] is False
    assert k.spec.requested_ensure_row_contiguous is True
    assert k.spec.ensure_row_contiguous is False


def test_contiguity_telemetry_tracks_copy_risk(monkeypatch) -> None:
    monkeypatch.delenv(_ENV_DISABLE, raising=False)
    monkeypatch.setenv(_ENV_TELEMETRY, "1")
    monkeypatch.setenv(_ENV_TELEMETRY_KERNELS, "kk_probe_*")

    k, _ = _build_kernel(monkeypatch, name="kk_probe_layout")
    x = np.arange(24, dtype=np.float32).reshape(4, 6).T  # non-row-contiguous

    out = k(x)

    assert out[0].shape == x.shape
    assert k.stats.contiguity_launches == 1
    assert k.stats.contiguity_checks == 1
    assert k.stats.contiguity_unknown_inputs == 0
    assert k.stats.non_row_contiguous_inputs == 1
    assert k.stats.launches_with_non_row_contiguous == 1
    assert k.stats.copy_risk_launches == 1
    assert k.stats.copy_risk_inputs == 1


def test_copy_risk_not_counted_when_row_contiguous_is_disabled(monkeypatch) -> None:
    monkeypatch.setenv(_ENV_DISABLE, "kk_probe_*")
    monkeypatch.setenv(_ENV_TELEMETRY, "1")
    monkeypatch.setenv(_ENV_TELEMETRY_KERNELS, "kk_probe_*")

    k, _ = _build_kernel(monkeypatch, name="kk_probe_layout")
    x = np.arange(24, dtype=np.float32).reshape(4, 6).T

    k(x)

    assert k.spec.requested_ensure_row_contiguous is True
    assert k.spec.ensure_row_contiguous is False
    assert k.stats.launches_with_non_row_contiguous == 1
    assert k.stats.copy_risk_launches == 0
    assert k.stats.copy_risk_inputs == 0
