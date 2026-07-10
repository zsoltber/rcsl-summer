"""
stage_a_mask_builder.py
=======================
On-device (Cortex-A53 / PYNQ) Stage-A scoring and mask construction for
integer-evolutionary tuning of a FINN-deployed QNN.

Pipeline implemented here:
  1. Load the STREAMLINED FINN ONNX model (standard MatMul + MultiThreshold
     ops, integer weights as initializers). NOT the post-partition HLS model.
  2. Run local (and optionally global) batches through it on the PS via
     qonnx's executor, harvesting the activation tensor entering each MatMul.
  3. Compute Stage-A metrics per layer:
        A1 headroom          (per weight)
        A2 activation energy (per input neuron, local data)
        A3 activation shift  (per input neuron, signed, z-normalised,
                              lever-gated, quadrant-classified)
  4. Build an "educated" boolean mask selecting p% of a chosen layer's
     weights, plus size-matched random control masks for the tournament.

Outputs a .npz your existing DE scripts can consume:
    mask            boolean, HW layer orientation (see TRANSPOSE_FOR_HW)
    random_masks    (X, ...) boolean control masks, same size
    scores          per-weight potency used for ranking
    quadrant        per-input-neuron quadrant code (0..3)
    E_local, E_global, z_shift, headroom   raw metric arrays

Everything the DE needs from this file is the mask; scoring is one-off.

Dependencies on the PYNQ image:  numpy, onnx, qonnx
    pip install qonnx        (pulls onnx; no torch needed)
"""

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx

# ----------------------------------------------------------------------------
# ------------------------- PLACEHOLDER CONFIGURATION ------------------------
# ----------------------------------------------------------------------------

MODEL_PATH        = "streamlined_model.onnx"   # streamlined FINN-ONNX twin
LOCAL_BATCH_PATH  = "local_batch.npy"          # (N_loc, n_features) quantized
GLOBAL_BATCH_PATH = "global_batch.npy"         # (N_glob, n_features) OR None
GLOBAL_STATS_PATH = None                       # precomputed stats .npz OR None
                                               # (ship this from the PC and set
                                               #  GLOBAL_BATCH_PATH = None to
                                               #  skip the global pass on-device)

BITWIDTH          = 4                          # weight bitwidth
INT_MAX           = 2 ** (BITWIDTH - 1) - 1    # +7 for INT4

TARGET_LAYER      = -1        # which MatMul layer to mask (index into the
                              # topologically ordered MatMul list; -1 = last)
MASK_FRACTION     = 0.10      # p% of the layer's weights
N_RANDOM_MASKS    = 10        # size-matched random controls for the tournament

# Gates / thresholds for eligibility (tune on first real score maps)
LEVER_EPS_PCTL    = 20.0      # E_local gate: neurons below this percentile of
                              # the nonzero-energy distribution count as silent
Z_MIN             = 2.0       # min |z| of shift to count a column as "shifted"
HEADROOM_MIN      = 1         # exclude weights with headroom < this (1 kills
                              # saturated weights; 0 disables the filter)
PER_NEURON_CAP    = 0.35      # max fraction of the mask on one output neuron

CHUNK             = 64        # samples per executor call (falls back to 1)

# CRITICAL ORIENTATION SWITCH -------------------------------------------------
# ONNX MatMul computes X @ W with W of shape (n_in, n_out): input neurons run
# along AXIS 0 of the initializer. Your pack_layer / .dat convention may store
# the layer as (n_out, n_in). If your DE-side arrays are (n_out, n_in), set
# this True so the saved mask is transposed to match. VERIFY on a layer with
# n_in != n_out before trusting any result.
TRANSPOSE_FOR_HW  = False

OUTPUT_PATH       = "stage_a_mask.npz"
RNG               = np.random.default_rng(1234)

# ----------------------------------------------------------------------------
# ------------------------- MODEL INTROSPECTION ------------------------------
# ----------------------------------------------------------------------------

def find_matmul_layers(model):
    """Return a topologically ordered list of dicts describing each MatMul:
    weight initializer name/array, the dynamic activation tensor feeding it,
    and its output (accumulator) tensor name."""
    layers = []
    init_names = {i.name for i in model.graph.initializer}
    for node in model.graph.node:                # graph.node is topo-ordered
        if node.op_type != "MatMul":
            continue
        w_name = next((c for c in node.input if c in init_names), None)
        if w_name is None:
            continue                             # dynamic-weight matmul: skip
        a_name = next(c for c in node.input if c != w_name)
        W = model.get_initializer(w_name)
        # Which axis of W indexes the INPUT neurons?
        # X @ W  -> W is input #1, shape (n_in, n_out), input axis 0.
        # W @ X  (rare after streamlining) -> input axis 1.
        input_axis = 0 if node.input[1] == w_name else 1
        layers.append(dict(node=node, w_name=w_name, a_name=a_name,
                           out_name=node.output[0], W=W,
                           input_axis=input_axis))
    if not layers:
        raise RuntimeError("No MatMul layers with initializer weights found. "
                           "Is this the streamlined model (not HLS-partitioned, "
                           "not raw QONNX with Quant nodes)?")
    return layers


def sanity_check_weights(layers):
    """Weights in the streamlined model must be integers within the INT range.
    If they are not, streamlining left scale factors inside the matmul path
    and NOTHING downstream (headroom, packing parity) can be trusted."""
    for li, L in enumerate(layers):
        W = L["W"]
        if not np.allclose(W, np.round(W)):
            raise RuntimeError(f"Layer {li}: non-integer weights "
                               f"(min={W.min()}, max={W.max()}). Streamlining "
                               "incomplete or wrong model exported.")
        if W.min() < -INT_MAX - 1 or W.max() > INT_MAX:
            raise RuntimeError(f"Layer {li}: weights outside INT{BITWIDTH} "
                               f"range [{-INT_MAX-1},{INT_MAX}]. Check BITWIDTH.")


# ----------------------------------------------------------------------------
# ------------------------- ACTIVATION HARVESTING ----------------------------
# ----------------------------------------------------------------------------

class NeuronStats:
    """Streaming per-neuron sum / sumsq of |a| (memory ~ 2 floats per neuron,
    never materialises the batch of activations)."""
    def __init__(self, n):
        self.n = 0
        self.s = np.zeros(n, dtype=np.float64)
        self.ss = np.zeros(n, dtype=np.float64)

    def update(self, A):                      # A: (chunk, n_neurons)
        a = np.abs(A.astype(np.float64))
        self.n += a.shape[0]
        self.s += a.sum(axis=0)
        self.ss += (a * a).sum(axis=0)

    def mean(self):
        return self.s / max(self.n, 1)

    def var(self):
        m = self.mean()
        return np.maximum(self.ss / max(self.n, 1) - m * m, 0.0)


def _exec_full(model, x, in_name):
    """One executor call returning the full tensor context."""
    return execute_onnx(model, {in_name: x.astype(np.float32)},
                        return_full_exec_context=True)


def harvest_stats(model, layers, X):
    """Run batch X through the model on the PS; return per-layer NeuronStats
    of the activations ENTERING each MatMul. Tries chunked execution first,
    falls back to per-sample if the graph has a hard batch-1 input shape."""
    in_name = model.graph.input[0].name
    stats = [NeuronStats(L["W"].shape[L["input_axis"]]) for L in layers]

    def consume(ctx):
        for L, st in zip(layers, stats):
            A = ctx[L["a_name"]]
            st.update(A.reshape(-1, st.s.shape[0]))

    i, step = 0, CHUNK
    while i < len(X):
        chunk = X[i:i + step]
        try:
            consume(_exec_full(model, chunk, in_name))
            i += step
        except Exception:
            if step == 1:
                raise
            step = 1                           # batch-1-only graph: go sample-wise
    return stats


# ----------------------------------------------------------------------------
# ------------------------- STAGE A METRICS ----------------------------------
# ----------------------------------------------------------------------------

def headroom_map(W):
    """Per-weight distance to the nearer clip wall. 0 = saturated."""
    return np.minimum(W - (-INT_MAX - 1), INT_MAX - W).astype(np.int32)
    # NOTE: if your convention is symmetric [-7,+7] (no -8), replace with
    #       np.minimum(W + INT_MAX, INT_MAX - W)


def shift_scores(E_loc, V_loc, n_loc, E_glob, V_glob, n_glob):
    """Signed shift, its z-score against pooled sampling error, and the
    quadrant code per input neuron:
        0 dead everywhere | 1 newly live | 2 deactivated | 3 live-changed"""
    delta = E_loc - E_glob
    se = np.sqrt(V_loc / max(n_loc, 1) + V_glob / max(n_glob, 1)) + 1e-12
    z = delta / se

    nz = E_loc[E_loc > 0]
    eps_loc = np.percentile(nz, LEVER_EPS_PCTL) if nz.size else 0.0
    nz = E_glob[E_glob > 0]
    eps_glob = np.percentile(nz, LEVER_EPS_PCTL) if nz.size else 0.0

    live_loc, live_glob = E_loc > eps_loc, E_glob > eps_glob
    quadrant = np.zeros_like(E_loc, dtype=np.int8)
    quadrant[ live_loc & ~live_glob] = 1        # newly live  -> best slots
    quadrant[~live_loc &  live_glob] = 2        # deactivated -> FREEZE
    quadrant[ live_loc &  live_glob] = 3        # live but (maybe) changed
    return delta, z, quadrant, live_loc


def build_layer_scores(L, st_loc, st_glob_mean, st_glob_var, n_loc, n_glob):
    """Combine A1-A3 into a per-weight potency array (ONNX orientation) and
    an eligibility mask implementing the mandatory lever gate."""
    W = L["W"]
    ax = L["input_axis"]

    E_loc, V_loc = st_loc.mean(), st_loc.var()
    delta, z, quadrant, live_loc = shift_scores(
        E_loc, V_loc, n_loc, st_glob_mean, st_glob_var, n_glob)

    # --- per-input-neuron column score ---------------------------------
    # shifted AND locally alive -> |z|; alive but unshifted -> small energy
    # tiebreak so eligible-but-unshifted weights can still fill the mask.
    energy_rank = E_loc / (E_loc.max() + 1e-12)          # in [0,1]
    col = np.where(live_loc & (np.abs(z) >= Z_MIN), np.abs(z),
                   np.where(live_loc, 0.01 * energy_rank, 0.0))
    col[quadrant == 2] = 0.0                              # hard freeze

    # --- broadcast to per-weight, apply headroom -------------------------
    shape = [1, 1]; shape[ax] = -1
    per_weight = np.broadcast_to(col.reshape(shape), W.shape).copy()
    hr = headroom_map(W)
    per_weight[hr < HEADROOM_MIN] = 0.0

    eligible = per_weight > 0.0
    return per_weight, eligible, dict(E_local=E_loc, E_global=st_glob_mean,
                                      delta=delta, z_shift=z,
                                      quadrant=quadrant, headroom=hr)


def assemble_mask(scores, eligible, W_shape, input_axis, frac, cap_frac, rng):
    """Top-k by score among eligible weights, with a per-output-neuron cap."""
    k = int(round(frac * scores.size))
    out_axis = 1 - input_axis
    n_out = W_shape[out_axis]
    cap = max(1, int(cap_frac * k))

    order = np.argsort(-scores, axis=None)
    mask = np.zeros(W_shape, dtype=bool)
    per_out = np.zeros(n_out, dtype=np.int32)
    taken = 0
    for flat in order:
        if taken == k:
            break
        idx = np.unravel_index(flat, W_shape)
        if not eligible[idx]:
            break                                # scores sorted: rest are 0
        o = idx[out_axis]
        if per_out[o] >= cap:
            continue
        mask[idx] = True
        per_out[o] += 1
        taken += 1

    if taken < k:                                # not enough eligible weights:
        pool = np.flatnonzero(~mask.ravel())     # honest fallback = random fill
        fill = rng.choice(pool, size=k - taken, replace=False)
        mask.ravel()[fill] = True
        print(f"[warn] only {taken}/{k} eligible; filled {k-taken} at random. "
              "Consider relaxing Z_MIN / HEADROOM_MIN or shrinking MASK_FRACTION.")
    return mask


def random_masks_like(mask, n_masks, rng):
    k = int(mask.sum())
    out = np.zeros((n_masks,) + mask.shape, dtype=bool)
    for m in range(n_masks):
        pick = rng.choice(mask.size, size=k, replace=False)
        out[m].ravel()[pick] = True
    return out


# ----------------------------------------------------------------------------
# ------------------------- FIDELITY CHECK (STUB) -----------------------------
# ----------------------------------------------------------------------------

def fidelity_check(model, X, pl_predict_fn, n=256):
    """MUST pass once before any score is trusted: the ONNX twin and the PL
    must produce identical predictions. pl_predict_fn(batch)->labels is your
    existing accelerator inference wrapper."""
    in_name = model.graph.input[0].name
    xs = X[:n]
    onnx_pred = []
    for i in range(0, len(xs), CHUNK):
        ctx = _exec_full(model, xs[i:i + CHUNK], in_name)
        out = ctx[model.graph.output[0].name]
        onnx_pred.append(np.argmax(out.reshape(out.shape[0], -1), axis=1))
    onnx_pred = np.concatenate(onnx_pred)
    pl_pred = pl_predict_fn(xs)
    agree = float(np.mean(onnx_pred == np.asarray(pl_pred)))
    print(f"[fidelity] ONNX vs PL agreement: {agree:.4f} on {len(xs)} samples")
    if agree < 1.0:
        raise RuntimeError("Emulator disagrees with fabric — do not proceed. "
                           "Wrong model file, stale weights, or wrong input "
                           "preprocessing/quantization are the usual causes.")


# ----------------------------------------------------------------------------
# ------------------------- MAIN ---------------------------------------------
# ----------------------------------------------------------------------------

def main():
    model = ModelWrapper(MODEL_PATH)
    layers = find_matmul_layers(model)
    sanity_check_weights(layers)
    print(f"[info] {len(layers)} MatMul layers; shapes: "
          f"{[L['W'].shape for L in layers]}")

    li = TARGET_LAYER % len(layers)
    L = layers[li]

    X_loc = np.load(LOCAL_BATCH_PATH)

    # --- global statistics: precomputed (preferred) or computed here -----
    if GLOBAL_STATS_PATH is not None:
        g = np.load(GLOBAL_STATS_PATH)
        gm, gv, n_glob = g[f"mean_{li}"], g[f"var_{li}"], int(g["n"])
        st_loc = harvest_stats(model, layers, X_loc)[li]
    elif GLOBAL_BATCH_PATH is not None:
        X_glob = np.load(GLOBAL_BATCH_PATH)
        st_loc = harvest_stats(model, layers, X_loc)[li]
        st_g = harvest_stats(model, layers, X_glob)[li]
        gm, gv, n_glob = st_g.mean(), st_g.var(), st_g.n
    else:
        raise RuntimeError("Provide GLOBAL_STATS_PATH or GLOBAL_BATCH_PATH.")

    # --- scores + mask -----------------------------------------------------
    scores, eligible, extras = build_layer_scores(
        L, st_loc, gm, gv, st_loc.n, n_glob)
    mask = assemble_mask(scores, eligible, L["W"].shape, L["input_axis"],
                         MASK_FRACTION, PER_NEURON_CAP, RNG)
    rand = random_masks_like(mask, N_RANDOM_MASKS, RNG)

    q = extras["quadrant"]
    print(f"[info] layer {li}: quadrants dead/new/deact/live = "
          f"{[(q == c).sum() for c in range(4)]}, "
          f"mask size {mask.sum()} ({100*MASK_FRACTION:.0f}%)")

    if TRANSPOSE_FOR_HW and L["input_axis"] == 0:
        mask_out = mask.T                       # -> (n_out, n_in) HW layout
        rand_out = rand.transpose(0, 2, 1)
        scores_out = scores.T
    else:
        mask_out, rand_out, scores_out = mask, rand, scores

    np.savez(OUTPUT_PATH, mask=mask_out, random_masks=rand_out,
             scores=scores_out, layer_index=li,
             onnx_weight_name=L["w_name"], **extras)
    print(f"[done] wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()