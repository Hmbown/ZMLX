from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

QuantType = Literal["none", "int2", "int4", "fp8"]


def bits_for_type(t: QuantType) -> int:
    return {"none": 0, "int2": 2, "int4": 4, "fp8": 8}[t]


@dataclass(frozen=True)
class GroupSpec:
    """A contiguous group of PCA coordinates to quantize together."""

    size: int
    qtype: QuantType  # determines bits per value

    def bits_per_value(self) -> int:
        return bits_for_type(self.qtype)

    def has_payload(self) -> bool:
        return self.bits_per_value() > 0

    def overhead_bits(self) -> int:
        # Per-group shift and scale, each stored as float16 (16 bits)
        return 32 if self.has_payload() else 0

    def payload_bits(self) -> int:
        return self.size * self.bits_per_value()

    def total_bits(self) -> int:
        return self.payload_bits() + self.overhead_bits()


@dataclass(frozen=True)
class QuantPlan:
    """A full plan covering the first `r` PCA coordinates."""

    groups: list[GroupSpec]

    def r(self) -> int:
        return sum(g.size for g in self.groups)

    def to_json(self) -> dict[str, Any]:
        return {"groups": [{"size": g.size, "qtype": g.qtype} for g in self.groups]}

    @staticmethod
    def from_json(obj: dict[str, Any]) -> QuantPlan:
        groups = [GroupSpec(size=int(g["size"]), qtype=str(g["qtype"])) for g in obj["groups"]]
        return QuantPlan(groups=groups)
