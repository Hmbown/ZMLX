from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runtime import (
    CompiledFusedElementwise,
    CompiledFusedReductionTwoPass,
    CompiledQuantizedSharedInputFusion,
    export_compiled_aot_payload,
)


@dataclass(frozen=True)
class FusedAOTArtifact:
    """Minimal AOT artifact scaffold for fused kernels."""

    descriptor: dict[str, Any]
    kernels: dict[str, str]
    sources: dict[str, str]


def build_aot_artifact(
    *,
    descriptor: dict[str, Any],
    compiled: CompiledFusedElementwise
    | CompiledFusedReductionTwoPass
    | CompiledQuantizedSharedInputFusion,
) -> FusedAOTArtifact:
    payload = export_compiled_aot_payload(descriptor=descriptor, compiled=compiled)
    return FusedAOTArtifact(
        descriptor=dict(payload["descriptor"]),
        kernels=dict(payload["kernels"]),
        sources=dict(payload["sources"]),
    )


def write_aot_artifact(
    artifact: FusedAOTArtifact,
    *,
    output_dir: str | Path,
    stem: str = "fused_kernel",
) -> Path:
    """Write descriptor JSON + MSL source files for offline packaging."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    descriptor_path = out_dir / f"{stem}.descriptor.json"
    descriptor_path.write_text(
        json.dumps(
            {
                "descriptor": artifact.descriptor,
                "kernels": artifact.kernels,
                "sources": sorted(artifact.sources.keys()),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    for source_name, source in artifact.sources.items():
        source_path = out_dir / f"{stem}.{source_name}.metal"
        source_path.write_text(source, encoding="utf-8")
    return descriptor_path


__all__ = [
    "FusedAOTArtifact",
    "build_aot_artifact",
    "write_aot_artifact",
]
