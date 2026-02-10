from __future__ import annotations

try:
    import numpy as np
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "KVTC requires numpy. Install with: pip install zmlx[kvtc]"
    ) from e


def packed_nbytes(n_values: int, bits: int) -> int:
    if bits <= 0:
        return 0
    return (n_values * bits + 7) // 8


def pack_uint(values: np.ndarray, bits: int) -> bytes:
    """Pack unsigned integers into a little-endian bitstream.

    values must be integer dtype, and each value must satisfy 0 <= v < 2**bits.
    """
    if bits == 0:
        return b""
    if bits == 8:
        return values.astype(np.uint8, copy=False).tobytes(order="C")

    mask = (1 << bits) - 1
    out = bytearray()
    bitbuf = 0
    bitcount = 0
    for v in values.astype(np.uint32, copy=False).tolist():
        bitbuf |= (int(v) & mask) << bitcount
        bitcount += bits
        while bitcount >= 8:
            out.append(bitbuf & 0xFF)
            bitbuf >>= 8
            bitcount -= 8
    if bitcount > 0:
        out.append(bitbuf & 0xFF)
    return bytes(out)


def unpack_uint(data: bytes, bits: int, n_values: int) -> tuple[np.ndarray, int]:
    """Unpack unsigned integers from a little-endian bitstream.

    Returns (values, nbytes_consumed).
    """
    if bits == 0:
        return np.zeros((n_values,), dtype=np.uint8), 0
    if bits == 8:
        need = n_values
        if len(data) < need:
            raise ValueError("not enough data for unpack_uint(bits=8)")
        return np.frombuffer(data[:need], dtype=np.uint8).copy(), need

    need = packed_nbytes(n_values, bits)
    if len(data) < need:
        raise ValueError(f"not enough data: need {need} bytes, got {len(data)}")
    buf = data[:need]

    mask = (1 << bits) - 1
    out = np.empty((n_values,), dtype=np.uint8)
    bitbuf = 0
    bitcount = 0
    idx = 0
    for b in buf:
        bitbuf |= int(b) << bitcount
        bitcount += 8
        while bitcount >= bits and idx < n_values:
            out[idx] = bitbuf & mask
            bitbuf >>= bits
            bitcount -= bits
            idx += 1
    if idx != n_values:
        raise ValueError("unpack_uint did not produce the requested number of values")
    return out, need
