from __future__ import annotations

from .base import KernelOp
from .dequantize import DequantizeOp
from .gather import GatherOp
from .grouped_gemm import GroupedGemmOp
from .kv_append import KVAppendOp
from .layernorm import LayerNormOp
from .moe_combine import MoECombineOp
from .moe_dispatch import MoEDispatchOp
from .moe_pack import MoEPackOp
from .moe_topk import MoETopKOp
from .quantize import QuantizeOp
from .rmsnorm import RMSNormOp
from .rope import RoPEOp
from .scatter import ScatterOp
from .softmax import SoftmaxOp
from .swiglu import SwiGLUOp
from .topk import TopKOp

_ALL_OPS = [
    RMSNormOp,
    SwiGLUOp,
    RoPEOp,
    SoftmaxOp,
    LayerNormOp,
    QuantizeOp,
    DequantizeOp,
    KVAppendOp,
    GatherOp,
    ScatterOp,
    TopKOp,
    MoETopKOp,
    MoEPackOp,
    MoEDispatchOp,
    MoECombineOp,
    GroupedGemmOp,
]


def get_registry() -> dict[str, KernelOp]:
    """Return a name -> KernelOp instance mapping for all registered ops."""
    return {cls.name: cls() for cls in _ALL_OPS}
