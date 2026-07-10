#!/usr/bin/env python3

"""
npy_to_finn_dat.py

Convert a layer weight matrix stored as a .npy file into FINN's
decoupled weight .dat format (what load_runtime_weights() reads
on the PYNQ board, or the $readmemh init file for the memstream).

Two ways to supply the layer metadata:

  A) From a post-folding intermediate ONNX model (safest, cross-checked):
     python npy_to_finn_dat.py \
         --model output/intermediate_models/step_apply_folding_config.onnx \
         --node 0 --npy w0.npy --out mvau_0.dat

  B) Standalone — feed the parameters directly, no model needed:
     python npy_to_finn_dat.py \
         --pe 4 --simd 8 --wdt INT2 \
         --npy w0.npy --out mvau_0.dat
     MW/MH are inferred from the .npy shape (rows=MW, cols=MH) unless
     given explicitly with --mw/--mh.

In both cases the actual packing is done by FINN's own
MVAU.make_weight_file(), so the output is bit-identical to what the
build flow generates. Run inside the FINN Docker environment.
"""

import argparse
import sys

import numpy as np
from onnx import helper
from finn.custom_op.registry import getCustomOp

# op_types that count as an MVAU across FINN versions
MVAU_OPTYPES = ("MVAU_hls", "MVAU_rtl", "MatrixVectorActivation", "MVAU")

# (op_type, domain) candidates for building a dummy node, newest first
DUMMY_NODE_CANDIDATES = [
    ("MVAU_hls", "finn.custom_op.fpgadataflow.hls"),
    ("MatrixVectorActivation", "finn.custom_op.fpgadataflow"),
]


def find_mvau_node(model, node_spec):
    """Return the MVAU node matching an index or a name."""
    mvau_nodes = [n for n in model.graph.node if n.op_type in MVAU_OPTYPES]
    if not mvau_nodes:
        sys.exit("ERROR: no MVAU nodes found in this model. "
                 "Make sure you are using a post-folding intermediate model.")
    try:
        idx = int(node_spec)
        if idx < 0 or idx >= len(mvau_nodes):
            sys.exit(f"ERROR: node index {idx} out of range "
                     f"(model has {len(mvau_nodes)} MVAU nodes).")
        return mvau_nodes[idx]
    except ValueError:
        pass
    for n in mvau_nodes:
        if n.name == node_spec:
            return n
    names = ", ".join(n.name for n in mvau_nodes)
    sys.exit(f"ERROR: no MVAU node named '{node_spec}'. Available: {names}")


def make_dummy_mvau(mw, mh, pe, simd, wdt_name):
    """Build an in-memory MVAU node carrying just the attributes that
    make_weight_file() needs, and return its FINN custom-op wrapper.
    No model / graph is required."""
    last_err = None
    for op_type, domain in DUMMY_NODE_CANDIDATES:
        try:
            node = helper.make_node(
                op_type,
                inputs=["in0", "weights"],
                outputs=["out0"],
                domain=domain,
                backend="fpgadataflow",
                MW=mw,
                MH=mh,
                SIMD=simd,
                PE=pe,
                weightDataType=wdt_name,
                # placeholders; not used in weight packing but some
                # FINN versions require them to be present:
                inputDataType="INT8",
                outputDataType="INT8",
                noActivation=1,
                ActVal=0,
                numInputVectors=[1],
                mem_mode="internal_decoupled",
                name=f"dummy_{op_type}_0",
            )
            inst = getCustomOp(node)
            # poke an attribute to confirm the wrapper resolved properly
            inst.get_nodeattr("MW")
            return inst
        except Exception as e:  # try the next op_type/domain pair
            last_err = e
    sys.exit(f"ERROR: could not construct a dummy MVAU node in this FINN "
             f"version. Last error: {last_err}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npy", required=True,
                    help="Input .npy weight matrix, shape (MW, MH)")
    ap.add_argument("--out", required=True,
                    help="Output .dat file path (e.g. mvau_0.dat)")
    ap.add_argument("--mode", default="decoupled_runtime",
                    choices=["decoupled_runtime", "decoupled_verilog_dat",
                             "decoupled_npy"],
                    help="Weight file flavour (default: decoupled_runtime)")
    # --- option A: model-based metadata ---
    ap.add_argument("--model", help="Post-folding intermediate ONNX model")
    ap.add_argument("--node", help="MVAU node index (int) or node name")
    # --- option B: direct parameters ---
    ap.add_argument("--pe", type=int, help="PE folding factor")
    ap.add_argument("--simd", type=int, help="SIMD folding factor")
    ap.add_argument("--wdt", help="Weight datatype name, e.g. INT2, INT4, UINT4, BIPOLAR")
    ap.add_argument("--mw", type=int,
                    help="Matrix width / input dim (default: npy rows)")
    ap.add_argument("--mh", type=int,
                    help="Matrix height / output dim (default: npy cols)")
    args = ap.parse_args()

    weights = np.load(args.npy)
    if weights.ndim != 2:
        sys.exit(f"ERROR: expected a 2D weight matrix, got shape {weights.shape}.")
    print(f"Loaded {args.npy}: shape={weights.shape} dtype={weights.dtype}")

    if args.model:
        # ---------- option A: pull everything from the model ----------
        if not args.node:
            sys.exit("ERROR: --node is required when using --model.")
        from qonnx.core.modelwrapper import ModelWrapper
        model = ModelWrapper(args.model)
        node = find_mvau_node(model, args.node)
        inst = getCustomOp(node)
        src = f"model node {node.name}"
    else:
        # ---------- option B: build a dummy node from CLI params ----------
        missing = [f for f in ("pe", "simd", "wdt") if getattr(args, f) is None]
        if missing:
            sys.exit("ERROR: without --model you must supply "
                     "--pe, --simd and --wdt "
                     f"(missing: {', '.join('--' + m for m in missing)}).")
        mw = args.mw if args.mw is not None else weights.shape[0]
        mh = args.mh if args.mh is not None else weights.shape[1]
        inst = make_dummy_mvau(mw, mh, args.pe, args.simd, args.wdt)
        src = "CLI parameters"

    mw = inst.get_nodeattr("MW")
    mh = inst.get_nodeattr("MH")
    pe = inst.get_nodeattr("PE")
    simd = inst.get_nodeattr("SIMD")
    wdt = inst.get_weight_datatype()
    print(f"Metadata from {src}:")
    print(f"  MW={mw} MH={mh}  PE={pe} SIMD={simd}  wdt={wdt.name} "
          f"({wdt.bitwidth()} bit)")

    # --- sanity checks ---
    if mh % pe != 0:
        sys.exit(f"ERROR: MH={mh} not divisible by PE={pe}.")
    if mw % simd != 0:
        sys.exit(f"ERROR: MW={mw} not divisible by SIMD={simd}.")

    if weights.shape != (mw, mh):
        if weights.shape == (mh, mw):
            print("NOTE: weights look transposed (MH, MW); transposing to (MW, MH).")
            weights = weights.T
        else:
            sys.exit(f"ERROR: weight shape {weights.shape} does not match "
                     f"(MW, MH) = ({mw}, {mh}).")

    if not (weights.min() >= wdt.min() and weights.max() <= wdt.max()):
        sys.exit(f"ERROR: weight values [{weights.min()}, {weights.max()}] fall "
                 f"outside the range of {wdt.name} [{wdt.min()}, {wdt.max()}]. "
                 "Quantise them first.")
    if not np.array_equal(weights, np.round(weights)):
        sys.exit(f"ERROR: weights contain non-integer values; {wdt.name} "
                 "requires values on the integer grid. Quantise them first.")

    weights = weights.astype(np.float32)  # FINN expects a float container

    # --- FINN does the folding + bit packing + 32-bit padding ---
    inst.make_weight_file(weights, args.mode, args.out)
    print(f"Wrote {args.out} ({args.mode})")


if __name__ == "__main__":
    main()