from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Literal, cast

try:
    import numpy as np
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "KVTC requires numpy. Install with: pip install zmlx[kvtc]"
    ) from e

from .bitpack import pack_uint, packed_nbytes, unpack_uint
from .plan import QuantPlan, bits_for_type
from .rope import RotaryConfig, RotaryEmbedding
from .utils import require, to_numpy

MAGIC = b"ZMLXKVTC"  # 8 bytes
VERSION = 1


@dataclass
class CalibrationArtifacts:
    """Calibration artifacts for KVTC-style compression.

    Stored on disk as:
      k_mu.npy, k_V.npy, k_plan.json
      v_mu.npy, v_V.npy, v_plan.json
      meta.json  (optional)
    """

    k_mu: np.ndarray
    k_V: np.ndarray
    k_plan: QuantPlan
    v_mu: np.ndarray
    v_V: np.ndarray
    v_plan: QuantPlan
    meta: dict[str, Any]

    @staticmethod
    def from_dir(path: str) -> CalibrationArtifacts:
        import os

        def load_json(p: str) -> dict[str, Any]:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(f"expected JSON object in {p}, got {type(data).__name__}")
            return cast(dict[str, Any], data)

        k_mu = np.load(os.path.join(path, "k_mu.npy"))
        k_V = np.load(os.path.join(path, "k_V.npy"))
        k_plan = QuantPlan.from_json(load_json(os.path.join(path, "k_plan.json")))

        v_mu = np.load(os.path.join(path, "v_mu.npy"))
        v_V = np.load(os.path.join(path, "v_V.npy"))
        v_plan = QuantPlan.from_json(load_json(os.path.join(path, "v_plan.json")))

        meta_path = os.path.join(path, "meta.json")
        meta = load_json(meta_path) if os.path.exists(meta_path) else {}

        return CalibrationArtifacts(
            k_mu=k_mu, k_V=k_V, k_plan=k_plan,
            v_mu=v_mu, v_V=v_V, v_plan=v_plan,
            meta=meta,
        )


def _as_4d(arr: np.ndarray) -> np.ndarray:
    """Normalize KV arrays to (batch, heads, seq, head_dim)."""
    if arr.ndim == 4:
        return arr
    if arr.ndim == 3:
        return arr[None, ...]
    raise ValueError(f"Expected KV tensor with 3 or 4 dims, got shape {arr.shape}")


def _pack_chunk(chunk: bytes) -> bytes:
    return struct.pack("<Q", len(chunk)) + chunk


def _unpack_chunk(buf: memoryview, offset: int) -> tuple[bytes, int]:
    if offset + 8 > len(buf):
        raise ValueError("corrupt blob: missing chunk length")
    (n,) = struct.unpack_from("<Q", buf, offset)
    offset += 8
    if offset + n > len(buf):
        raise ValueError("corrupt blob: chunk exceeds blob length")
    data = bytes(buf[offset : offset + n])
    offset += n
    return data, offset


def _quantize_row_uniform(row: np.ndarray, plan: QuantPlan) -> bytes:
    """Quantize a single PCA coefficient row (r,) following the plan."""
    out = bytearray()
    idx = 0
    for g in plan.groups:
        bits = bits_for_type(g.qtype)
        seg = row[idx : idx + g.size]
        idx += g.size
        if bits == 0:
            continue

        levels = 1 << bits
        vmin = float(seg.min()) if seg.size else 0.0
        vmax = float(seg.max()) if seg.size else 0.0
        shift = vmin
        scale = (vmax - vmin) / (levels - 1) if vmax > vmin else 1.0

        q = np.rint((seg - shift) / scale).astype(np.int32)
        q = np.clip(q, 0, levels - 1).astype(np.uint8)

        out += np.float16(shift).tobytes()
        out += np.float16(scale).tobytes()
        out += pack_uint(q, bits)
    return bytes(out)


def _dequantize_row_uniform(data: bytes, plan: QuantPlan) -> tuple[np.ndarray, int]:
    """Decode one quantized row into PCA coordinates."""
    r = plan.r()
    out = np.zeros((r,), dtype=np.float32)
    idx = 0
    off = 0
    mv = memoryview(data)

    for g in plan.groups:
        bits = bits_for_type(g.qtype)
        if bits == 0:
            idx += g.size
            continue

        if off + 4 > len(mv):
            raise ValueError("corrupt payload: missing shift/scale")
        shift = np.frombuffer(mv[off : off + 2], dtype=np.float16)[0].astype(np.float32)
        scale = np.frombuffer(mv[off + 2 : off + 4], dtype=np.float16)[0].astype(np.float32)
        off += 4

        nbytes = packed_nbytes(g.size, bits)
        q, used = unpack_uint(bytes(mv[off : off + nbytes]), bits, g.size)
        off += used

        out[idx : idx + g.size] = q.astype(np.float32) * scale + shift
        idx += g.size

    return out, off


def _payload_nbytes_per_row(plan: QuantPlan) -> int:
    n = 0
    for g in plan.groups:
        bits = bits_for_type(g.qtype)
        if bits == 0:
            continue
        n += 4  # shift+scale float16
        n += packed_nbytes(g.size, bits)
    return n


def _flatten_kv_mid(
    kv_layers: list[np.ndarray],
    positions: np.ndarray,
    rope: RotaryEmbedding | None,
    apply_rope: bool,
) -> np.ndarray:
    """Flatten KV at given token positions into X (n_pos, p)."""
    n_pos = int(positions.shape[0])
    parts = []
    for layer_kv in kv_layers:
        a = _as_4d(layer_kv).astype(np.float16, copy=False)
        b, h, t, d = a.shape
        require(b == 1, "This scaffold currently assumes batch=1 KV caches.")
        seg = np.take(a[0], positions, axis=1).astype(np.float32)  # (h, n, d)
        if apply_rope and rope is not None:
            seg = rope.apply(seg, positions, inverse=True)
        seg2 = np.transpose(seg, (1, 0, 2)).reshape(n_pos, h * d)
        parts.append(seg2.astype(np.float16))
    return np.concatenate(parts, axis=1) if parts else np.zeros((n_pos, 0), dtype=np.float16)


def _unflatten_kv_mid(
    X: np.ndarray,
    template_layers: list[np.ndarray],
    positions: np.ndarray,
    rope: RotaryEmbedding | None,
    apply_rope: bool,
) -> list[np.ndarray]:
    """Inverse of _flatten_kv_mid."""
    n_pos = int(positions.shape[0])
    out_layers: list[np.ndarray] = []
    offset = 0
    for tmpl in template_layers:
        a = _as_4d(tmpl).astype(np.float16, copy=False)
        b, h, t, d = a.shape
        require(b == 1, "This scaffold currently assumes batch=1 KV caches.")
        width = h * d
        seg = X[:, offset : offset + width]
        offset += width
        seg = seg.astype(np.float32).reshape(n_pos, h, d).transpose(1, 0, 2)
        if apply_rope and rope is not None:
            seg = rope.apply(seg, positions, inverse=False)
        out = a.copy()
        out[0, :, positions, :] = np.transpose(seg, (1, 0, 2)).astype(np.float16)
        out_layers.append(out)
    return out_layers


class KVTCCacheCodec:
    """KVTC-style KV cache compressor/decompressor.

    Supports two modes:
    - ``"dual_stream"`` (default): Separate K and V caches, both compressed.
    - ``"single_stream"``: Keys-only cache (e.g. GLM MLA). V-side compression
      is skipped entirely. The blob metadata records the mode.
    """

    def __init__(
        self,
        artifacts: CalibrationArtifacts,
        w: int = 128,
        s: int = 4,
        rope_cfg: RotaryConfig | None = None,
        apply_rope_to_keys: bool = True,
        apply_rope_to_values: bool = False,
        zlib_level: int = 3,
        chunk_rows: int = 256,
        mode: Literal["dual_stream", "single_stream"] = "dual_stream",
    ):
        self.artifacts = artifacts
        self.w = int(w)
        self.s = int(s)
        self.zlib_level = int(zlib_level)
        self.chunk_rows = int(chunk_rows)
        self.mode = mode

        self.apply_rope_to_keys = bool(apply_rope_to_keys)
        self.apply_rope_to_values = bool(apply_rope_to_values)

        self.rope = RotaryEmbedding(rope_cfg) if rope_cfg is not None else None

    @staticmethod
    def from_calibration_dir(
        path: str,
        w: int = 128,
        s: int = 4,
        rope_cfg: RotaryConfig | None = None,
        apply_rope_to_keys: bool = True,
        apply_rope_to_values: bool = False,
        zlib_level: int = 3,
        chunk_rows: int = 256,
        mode: Literal["dual_stream", "single_stream"] = "dual_stream",
    ) -> KVTCCacheCodec:
        return KVTCCacheCodec(
            artifacts=CalibrationArtifacts.from_dir(path),
            w=w, s=s,
            rope_cfg=rope_cfg,
            apply_rope_to_keys=apply_rope_to_keys,
            apply_rope_to_values=apply_rope_to_values,
            zlib_level=zlib_level,
            chunk_rows=chunk_rows,
            mode=mode,
        )

    def compress(self, k_layers: list[Any], v_layers: list[Any]) -> bytes:
        """Compress a KV cache into a single blob (bytes).

        In single_stream mode, v_layers should be a list of empty/zero arrays
        (they are stored but not PCA-compressed).
        """
        k_np = [to_numpy(k) for k in k_layers]
        v_np = [to_numpy(v) for v in v_layers]
        require(len(k_np) == len(v_np), "K and V must have same number of layers")

        L = len(k_np)
        b, h, t, d = _as_4d(k_np[0]).shape
        require(b == 1, "This scaffold currently assumes batch=1 KV caches.")

        if self.mode == "dual_stream":
            # Validate all K and V layers share shape
            for layer_idx in range(L):
                kb, kh, kt, kd = _as_4d(k_np[layer_idx]).shape
                vb, vh, vt, vd = _as_4d(v_np[layer_idx]).shape
                require((kb, kh, kt, kd) == (b, h, t, d), "All K layers must share shape")
                require((vb, vh, vt, vd) == (b, h, t, d), "All V layers must share shape")
        else:
            # single_stream: only validate K layers
            for layer_idx in range(L):
                kb, kh, kt, kd = _as_4d(k_np[layer_idx]).shape
                require((kb, kh, kt, kd) == (b, h, t, d), "All K layers must share shape")

        require(t >= self.s + self.w, f"seq_len {t} too small for s={self.s}, w={self.w}")

        # Verify calibration dimensionality for keys
        p_k = L * h * d
        require(
            self.artifacts.k_mu.shape[0] == p_k,
            f"k_mu dim mismatch: expected {p_k}, got {self.artifacts.k_mu.shape[0]}",
        )
        require(
            self.artifacts.k_V.shape[0] == p_k,
            f"k_V dim mismatch: expected {p_k}, got {self.artifacts.k_V.shape[0]}",
        )
        require(
            self.artifacts.k_V.shape[1] == self.artifacts.k_plan.r(),
            "k_V rank must match plan.r()",
        )

        if self.mode == "dual_stream":
            p_v = L * h * d
            require(
                self.artifacts.v_mu.shape[0] == p_v,
                f"v_mu dim mismatch: expected {p_v}, got {self.artifacts.v_mu.shape[0]}",
            )
            require(
                self.artifacts.v_V.shape[0] == p_v,
                f"v_V dim mismatch: expected {p_v}, got {self.artifacts.v_V.shape[0]}",
            )
            require(
                self.artifacts.v_V.shape[1] == self.artifacts.v_plan.r(),
                "v_V rank must match plan.r()",
            )

        positions_mid = np.arange(self.s, t - self.w, dtype=np.int64)
        n_mid = int(positions_mid.shape[0])

        # Prefix/suffix raw storage (keys)
        prefix_k = np.stack(
            [_as_4d(x)[:, :, : self.s, :] for x in k_np], axis=0
        ).astype(np.float16)
        suffix_k = np.stack(
            [_as_4d(x)[:, :, t - self.w : t, :] for x in k_np], axis=0
        ).astype(np.float16)

        # Middle keys: flatten -> project -> quantize -> zlib
        mid_k_payload = b""
        if n_mid > 0:
            Xk = _flatten_kv_mid(k_np, positions_mid, self.rope, self.apply_rope_to_keys)
            mid_k_payload = self._encode_matrix(
                Xk, self.artifacts.k_mu, self.artifacts.k_V, self.artifacts.k_plan
            )

        # Values handling depends on mode
        if self.mode == "dual_stream":
            prefix_v = np.stack(
                [_as_4d(x)[:, :, : self.s, :] for x in v_np], axis=0
            ).astype(np.float16)
            suffix_v = np.stack(
                [_as_4d(x)[:, :, t - self.w : t, :] for x in v_np], axis=0
            ).astype(np.float16)

            mid_v_payload = b""
            if n_mid > 0:
                Xv = _flatten_kv_mid(
                    v_np, positions_mid, self.rope, self.apply_rope_to_values
                )
                mid_v_payload = self._encode_matrix(
                    Xv, self.artifacts.v_mu, self.artifacts.v_V, self.artifacts.v_plan
                )
        else:
            # single_stream: empty V data
            prefix_v = np.zeros((L, b, h, self.s, 0), dtype=np.float16)
            suffix_v = np.zeros((L, b, h, self.w, 0), dtype=np.float16)
            mid_v_payload = b""

        v_head_dim = int(prefix_v.shape[-1]) if prefix_v.ndim == 5 else d

        meta = {
            "version": VERSION,
            "layers": L,
            "batch": b,
            "heads": h,
            "seq": t,
            "head_dim": d,
            "v_head_dim": v_head_dim,
            "w": self.w,
            "s": self.s,
            "dtype": "float16",
            "n_mid": n_mid,
            "mode": self.mode,
            "payload_bytes_per_row_k": _payload_nbytes_per_row(self.artifacts.k_plan),
            "payload_bytes_per_row_v": _payload_nbytes_per_row(self.artifacts.v_plan),
            "rope": None if self.rope is None else {
                "dim": self.rope.cfg.dim,
                "base": self.rope.cfg.base,
                "traditional": self.rope.cfg.traditional,
                "offset": self.rope.cfg.offset,
            },
            "apply_rope_to_keys": self.apply_rope_to_keys,
            "apply_rope_to_values": self.apply_rope_to_values,
            "artifacts_meta": self.artifacts.meta,
        }
        meta_bytes = json.dumps(meta).encode("utf-8")

        header = struct.pack("<8sHHI", MAGIC, VERSION, 0, len(meta_bytes)) + meta_bytes

        blob = bytearray()
        blob += header
        blob += _pack_chunk(prefix_k.tobytes(order="C"))
        blob += _pack_chunk(suffix_k.tobytes(order="C"))
        blob += _pack_chunk(zlib.compress(mid_k_payload, level=self.zlib_level))

        blob += _pack_chunk(prefix_v.tobytes(order="C"))
        blob += _pack_chunk(suffix_v.tobytes(order="C"))
        blob += _pack_chunk(zlib.compress(mid_v_payload, level=self.zlib_level))

        return bytes(blob)

    def decompress(self, blob: bytes) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Decompress a blob back into per-layer K/V arrays (NumPy float16).

        In single_stream mode, v_layers will be a list of zero-filled arrays
        with v_head_dim=0.
        """
        mv = memoryview(blob)
        if len(mv) < struct.calcsize("<8sHHI"):
            raise ValueError("blob too small")
        magic, version, _reserved, meta_len = struct.unpack_from("<8sHHI", mv, 0)
        if magic != MAGIC:
            raise ValueError("not a ZMLXKVTC blob")
        if version != VERSION:
            raise ValueError(f"unsupported blob version {version}")

        off = struct.calcsize("<8sHHI")
        meta_bytes = bytes(mv[off : off + meta_len])
        off += meta_len
        meta = json.loads(meta_bytes.decode("utf-8"))

        L = int(meta["layers"])
        b = int(meta["batch"])
        h = int(meta["heads"])
        t = int(meta["seq"])
        d = int(meta["head_dim"])
        v_d = int(meta.get("v_head_dim", d))
        w = int(meta["w"])
        s = int(meta["s"])
        n_mid = int(meta["n_mid"])
        blob_mode = meta.get("mode", "dual_stream")

        require(b == 1, "This scaffold currently assumes batch=1 KV caches.")
        require(
            w == self.w and s == self.s,
            "Codec configured with different w/s than blob.",
        )

        prefix_k_bytes, off = _unpack_chunk(mv, off)
        suffix_k_bytes, off = _unpack_chunk(mv, off)
        mid_k_comp, off = _unpack_chunk(mv, off)

        prefix_v_bytes, off = _unpack_chunk(mv, off)
        suffix_v_bytes, off = _unpack_chunk(mv, off)
        mid_v_comp, off = _unpack_chunk(mv, off)

        # Reconstruct keys
        prefix_k = np.frombuffer(prefix_k_bytes, dtype=np.float16).reshape((L, b, h, s, d))
        suffix_k = np.frombuffer(suffix_k_bytes, dtype=np.float16).reshape((L, b, h, w, d))

        tmpl = [np.zeros((b, h, t, d), dtype=np.float16) for _ in range(L)]
        for layer_idx in range(L):
            tmpl[layer_idx][:, :, :s, :] = prefix_k[layer_idx]
            tmpl[layer_idx][:, :, t - w : t, :] = suffix_k[layer_idx]

        positions_mid = np.arange(s, t - w, dtype=np.int64)
        if n_mid > 0:
            mid_k_payload = zlib.decompress(mid_k_comp)
            Xk = self._decode_matrix(
                mid_k_payload, n_mid,
                self.artifacts.k_mu, self.artifacts.k_V, self.artifacts.k_plan,
            )
            k_layers = _unflatten_kv_mid(
                Xk, tmpl, positions_mid, self.rope, self.apply_rope_to_keys
            )
        else:
            k_layers = tmpl

        # Reconstruct values
        if blob_mode == "single_stream":
            # V arrays are zero-filled with v_head_dim=0
            v_layers = [np.zeros((b, h, t, v_d), dtype=np.float16) for _ in range(L)]
        else:
            prefix_v = np.frombuffer(
                prefix_v_bytes, dtype=np.float16
            ).reshape((L, b, h, s, v_d))
            suffix_v = np.frombuffer(
                suffix_v_bytes, dtype=np.float16
            ).reshape((L, b, h, w, v_d))

            tmpl_v = [np.zeros((b, h, t, v_d), dtype=np.float16) for _ in range(L)]
            for layer_idx in range(L):
                tmpl_v[layer_idx][:, :, :s, :] = prefix_v[layer_idx]
                tmpl_v[layer_idx][:, :, t - w : t, :] = suffix_v[layer_idx]

            if n_mid > 0:
                mid_v_payload = zlib.decompress(mid_v_comp)
                Xv = self._decode_matrix(
                    mid_v_payload, n_mid,
                    self.artifacts.v_mu, self.artifacts.v_V, self.artifacts.v_plan,
                )
                v_layers = _unflatten_kv_mid(
                    Xv, tmpl_v, positions_mid, self.rope, self.apply_rope_to_values
                )
            else:
                v_layers = tmpl_v

        return k_layers, v_layers

    def _encode_matrix(
        self, X: np.ndarray, mu: np.ndarray, V: np.ndarray, plan: QuantPlan
    ) -> bytes:
        """Project and quantize X (n, p) into a packed payload."""
        X = np.asarray(X, dtype=np.float16)
        n, p = X.shape
        require(mu.shape[0] == p and V.shape[0] == p, "PCA artifacts shape mismatch")
        r = plan.r()
        require(V.shape[1] == r, "V rank must match plan.r()")

        mu32 = mu.astype(np.float32)
        V32 = V.astype(np.float32)

        out = bytearray()
        for start in range(0, n, self.chunk_rows):
            end = min(n, start + self.chunk_rows)
            Xc = X[start:end].astype(np.float32)
            D = (Xc - mu32) @ V32  # (chunk, r)
            for i in range(D.shape[0]):
                out += _quantize_row_uniform(D[i], plan)
        return bytes(out)

    def _decode_matrix(
        self, payload: bytes, n_rows: int,
        mu: np.ndarray, V: np.ndarray, plan: QuantPlan,
    ) -> np.ndarray:
        """Decode a packed payload back into X (n, p)."""
        r = plan.r()
        p = mu.shape[0]
        require(V.shape[0] == p and V.shape[1] == r, "PCA artifacts shape mismatch")

        mu32 = mu.astype(np.float32)
        Vt32 = V.astype(np.float32).T  # (r, p)

        X_out = np.zeros((n_rows, p), dtype=np.float16)

        off = 0
        for i in range(n_rows):
            row_hat, used = _dequantize_row_uniform(payload[off:], plan)
            off += used
            x = (row_hat @ Vt32) + mu32
            X_out[i] = x.astype(np.float16)

        if off != len(payload):
            raise ValueError(
                f"payload length mismatch: consumed {off} bytes, "
                f"payload has {len(payload)} bytes"
            )
        return X_out
