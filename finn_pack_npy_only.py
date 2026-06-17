#!/usr/bin/env python3
"""
finn_pack_numpy_callable.py  --  pure-numpy FINN decoupled weight packer
                                 (callable / importable API, no CLI)

Same validated packing internals as finn_pack_numpy.py (ported from FINN
source and verified byte-identical to MVAU.make_weight_file across INT2/3/4/8,
UINT4, BIPOLAR and both decoupled modes). This variant exposes ONE primary
function you call from your own code:

    from finn_pack_numpy_callable import pack_layer

    pack_layer(npy_layer, out_path, pe, simd, wdt,
               mw=None, mh=None, mode="decoupled_runtime")

`npy_layer` may be either a path to a .npy file OR an in-memory numpy array
(handy in an evolution loop where weights never touch disk). mw/mh default to
the array's (rows, cols). Returns a small dict of the resolved parameters.

Depends ONLY on numpy, so it runs on a 32-bit PYNQ-Z2 (armv7l).
"""

import math
import textwrap
import numpy as np


# ----------------------------------------------------------------------
# minimal DataType description (integer + bipolar; covers quantised NN weights)
# ----------------------------------------------------------------------
class WDType:
    def __init__(self, name):
        self.name = name.upper()
        n = self.name
        if n == "BINARY":
            self._bw, self._signed, self._bipolar = 1, False, False
        elif n == "BIPOLAR":
            self._bw, self._signed, self._bipolar = 1, True, True
        elif n.startswith("UINT"):
            self._bw, self._signed, self._bipolar = int(n[4:]), False, False
        elif n.startswith("INT"):
            self._bw, self._signed, self._bipolar = int(n[3:]), True, False
        else:
            raise ValueError(f"unsupported weight datatype: {name}")

    def bitwidth(self):
        return self._bw

    def signed(self):
        return self._signed

    def is_bipolar(self):
        return self._bipolar

    def min(self):
        if self._bipolar:
            return -1
        if self._signed:
            return -(2 ** (self._bw - 1))
        return 0

    def max(self):
        if self._bipolar:
            return 1
        if self._signed:
            return 2 ** (self._bw - 1) - 1
        return 2 ** self._bw - 1

    def allowed(self, v):
        return self.min() <= v <= self.max()


# ----------------------------------------------------------------------
# packing + folding primitives (verbatim-faithful port of FINN)
# ----------------------------------------------------------------------
def _array2hexstring(array, dtype: WDType, pad_to_nbits, reverse=False):
    if pad_to_nbits < 4:
        pad_to_nbits = 4
    array = np.asarray(array, dtype=np.float32)
    assert array.ndim == 1, "array must be 1-D"
    if reverse:
        array = np.flip(array, -1)
    if dtype.is_bipolar():
        array = (array + 1) / 2
        bw, signed = 1, False
    else:
        bw, signed = dtype.bitwidth(), dtype.signed()
    bits, nbits, mask = 0, 0, (1 << bw) - 1
    for val in array:
        iv = int(round(float(val)))
        assert dtype.allowed(iv), f"value {iv} not permitted by {dtype.name}"
        iv = (iv + (1 << bw)) & mask if (signed and iv < 0) else (iv & mask)
        bits = (bits << bw) | iv
        nbits += bw
    if pad_to_nbits < nbits:
        raise Exception("Number of bits is greater than pad_to_nbits")
    return format(bits, "0{}x".format(pad_to_nbits // 4))


def _pack_innermost(ndarray, dtype, pad_to_nbits):
    ndarray = np.asarray(ndarray, dtype=np.float32)
    return np.apply_along_axis(
        lambda x: _array2hexstring(x, dtype, pad_to_nbits, reverse=False),
        axis=-1, arr=ndarray,
    )


def _interleave_outer(matrix, n_partitions):
    matrix = np.asarray(matrix, dtype=np.float32)
    shp = matrix.shape
    assert matrix.ndim == 2 and shp[0] % n_partitions == 0
    m = matrix.reshape(-1, n_partitions, shp[1]).transpose((1, 0, 2))
    return m.reshape(n_partitions, -1, shp[1])


def _hw_compatible_weight_tensor(W, mw, mh, pe, simd, dtype: WDType):
    assert W.shape == (mw, mh), f"weights must be (MW,MH)=({mw},{mh}), got {W.shape}"
    assert mw % simd == 0, "MW must be divisible by SIMD"
    assert mh % pe == 0, "MH must be divisible by PE"
    wmem = mw * mh // (pe * simd)
    ret = W.T
    if dtype.is_bipolar():
        ret = (ret + 1) / 2
    ret = _interleave_outer(ret, pe)
    ret = ret.reshape(1, pe, wmem, simd)
    ret = np.flip(ret, axis=-1)
    return ret


# ----------------------------------------------------------------------
# PRIMARY CALLABLE
# ----------------------------------------------------------------------
def pack_layer(npy_layer, out_path, pe, simd, wdt,
               mw=None, mh=None, mode="decoupled_runtime"):
    """Pack one layer's weights into FINN's decoupled .dat format.

    Parameters
    ----------
    npy_layer : str | numpy.ndarray
        Path to a .npy file, OR an in-memory (MW, MH) integer-valued array.
    out_path  : str
        Destination .dat path.
    pe, simd  : int
        Folding factors for this MVAU (from the build's step_hls_codegen.onnx).
    wdt       : str
        Weight datatype name, e.g. "INT4", "INT2", "UINT4", "BIPOLAR".
    mw, mh    : int, optional
        Matrix width/height. Default to the array's (rows, cols).
    mode      : str
        "decoupled_runtime" (board runtime weights, 32-bit words) or
        "decoupled_verilog_dat" ($readmemh full-width words).

    Returns
    -------
    dict with the resolved {mw, mh, pe, simd, wdt, mode, out_path}.
    """
    W = np.load(npy_layer) if isinstance(npy_layer, str) else np.asarray(npy_layer)
    W = np.asarray(W, dtype=np.float32)
    if W.ndim != 2:
        raise ValueError(f"expected a 2-D weight matrix, got shape {W.shape}")

    mw = W.shape[0] if mw is None else mw
    mh = W.shape[1] if mh is None else mh

    # accept a transposed array, as the original script did
    if W.shape != (mw, mh) and W.shape == (mh, mw):
        W = W.T

    dtype = WDType(wdt)
    if not (W.min() >= dtype.min() and W.max() <= dtype.max()):
        raise ValueError(
            f"weights [{W.min()},{W.max()}] outside {dtype.name} "
            f"range [{dtype.min()},{dtype.max()}]"
        )
    if not np.array_equal(W, np.round(W)):
        raise ValueError("weights must be integer-valued for an integer datatype")

    wt = _hw_compatible_weight_tensor(W, mw, mh, pe, simd, dtype)
    wt = np.transpose(wt, (0, 2, 1, 3))            # (1,WMEM,PE,SIMD)
    wt_pe_flipped = np.flip(wt, axis=-2).reshape(1, -1, pe * simd).copy()

    bw = 1 if dtype.is_bipolar() else dtype.bitwidth()
    weight_width = pe * simd * bw
    pack_dtype = WDType("BINARY") if dtype.is_bipolar() else dtype

    if mode == "decoupled_verilog_dat":
        wwp = ((weight_width + 3) // 4) * 4
        packed = _pack_innermost(wt_pe_flipped, pack_dtype, wwp)
        with open(out_path, "w") as f:
            for v in packed.flatten():
                f.write(v + "\n")
    elif mode == "decoupled_runtime":
        wpm = max(2 ** math.ceil(math.log2(weight_width / 32)), 1)
        wwp = wpm * 32
        packed = _pack_innermost(wt_pe_flipped, pack_dtype, wwp)
        with open(out_path, "w") as f:
            for v in packed.flatten():
                words_32b = textwrap.wrap(v, 8)
                words_32b.reverse()
                for w in words_32b:
                    f.write(w + "\n")
    else:
        raise ValueError(f"unsupported mode: {mode}")

    return {"mw": mw, "mh": mh, "pe": pe, "simd": simd,
            "wdt": dtype.name, "mode": mode, "out_path": out_path}


def compare_to_reference(generated_path, reference_path):
    """One-time validation helper: returns (identical: bool, first_diff: str|None).
    Ignores 0x prefixes, case, and surrounding whitespace."""
    def norm(p):
        out = []
        with open(p) as f:
            for ln in f:
                s = ln.strip().lower()
                out.append(s[2:] if s.startswith("0x") else s)
        return out
    a, b = norm(generated_path), norm(reference_path)
    if len(a) != len(b):
        return False, f"line count differs: {len(a)} vs {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return False, f"line {i}: '{x}' != '{y}'"
    return True, None