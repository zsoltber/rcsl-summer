"""
Gradient-Free Training (GFT) — LOW-MEMORY refactor
===================================================

Same algorithm as gft_algorithms.py, but restructured so that the large
(B x d_in x d_out) contribution tensor is NEVER materialised. This matters
on constrained hardware such as the PYNQ-Z2, where holding a full
contribution tensor for even one layer can exhaust available memory.

Two changes, both mathematically identical to the original:

  1. Forward pass no longer stores C. Since v = sum_i C[b,i,o] = (X_prev @ W),
     we compute v directly with a matmul. The backward pass recomputes the
     per-column contribution slices on the fly from X_prev and W.

  2. Backward pass processes output features in CHUNKS. The B matrix sums
     over the batch independently for each output feature o, so chunking
     over o produces bit-identical results. Peak temporary memory drops
     from (B x d_in x d_out) to (B x d_in x chunk_size).

Set `out_chunk` small (e.g. 8-32) on tight memory; larger is faster.
"""

import numpy as np
import os

# ─────────────────────────────────────────────
#  Activation functions (unchanged)
# ─────────────────────────────────────────────

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def cross_entropy_loss(logits, labels):
    probs = softmax(logits)
    B = len(labels)
    loss = -np.mean(np.log(probs[np.arange(B), labels] + 1e-9))
    delta = probs.copy()
    delta[np.arange(B), labels] -= 1
    delta /= B
    return loss, delta


# ─────────────────────────────────────────────
#  Forward pass — NO contribution tensor stored
# ─────────────────────────────────────────────

def forward_lowmem(X_batch, weights, activations):
    """
    Standard forward pass. Stores only the per-layer activations X^(l),
    which the backward pass needs as the layer inputs. The contribution
    tensor C is never built — v is computed directly as X_prev @ W.

    Returns
    -------
    layer_outputs : list of L arrays, layer_outputs[l] = X^(l)  (B, d_l)
    """
    layer_outputs = []
    X_prev = X_batch
    for W, sigma in zip(weights, activations):
        v = X_prev @ W          # (B, d_l) — same as sum_i C[b,i,o]
        X_curr = sigma(v)
        layer_outputs.append(X_curr)
        X_prev = X_curr
    return layer_outputs


# ─────────────────────────────────────────────
#  Algorithm 3 — DynamicProbability (unchanged)
# ─────────────────────────────────────────────

def dynamic_probability(B_topk, p_min=0.001):
    abs_B = np.abs(B_topk)
    b_max = abs_B.max()
    b_min = abs_B[abs_B > 0].min() if (abs_B > 0).any() else 0
    if b_max == 0:
        return np.zeros_like(B_topk, dtype=float)
    if b_max == b_min:
        P = np.where(abs_B > 0, p_min, 0.0)
    else:
        P = np.where(abs_B > 0, (abs_B - b_min) / (b_max - b_min), 0.0)
        P = np.where(P > 0, np.clip(P, p_min, 1.0), 0.0)
    return P


# ─────────────────────────────────────────────
#  B-matrix computation — CHUNKED over output features
# ─────────────────────────────────────────────

def compute_B_matrix(delta, X_prev, W, out_chunk):
    """
    Compute B^(l) without ever holding a full (B, d_in, d_out) tensor.

    For each output-feature chunk [o0:o1]:
        C_slice   = X_prev[:, :, None] * W[None, :, o0:o1]   (B, d_in, chunk)
        err_mask  = (delta[:, None, o0:o1] * C_slice) < 0
        vote      = sign(delta[:, None, o0:o1] * X_prev[:, :, None])
        B[:, o0:o1] = (err_mask * vote).sum(axis=0)

    Peak temporary memory: (B, d_in, out_chunk) instead of (B, d_in, d_out).
    Result is identical to the unchunked computation.
    """
    d_in, d_out = W.shape
    B_mat = np.zeros((d_in, d_out))

    X_prev_3d = X_prev[:, :, np.newaxis]            # (B, d_in, 1) — reused

    for o0 in range(0, d_out, out_chunk):
        o1 = min(o0 + out_chunk, d_out)

        # delta slice for this chunk of outputs: (B, 1, chunk)
        delta_slice = delta[:, np.newaxis, o0:o1]

        # contribution slice: (B, d_in, chunk)
        C_slice = X_prev_3d * W[np.newaxis, :, o0:o1]

        # error mask: delta and contribution disagree in sign
        err_mask = (delta_slice * C_slice) < 0      # (B, d_in, chunk)

        # vote direction (independent of W's current value)
        vote = np.sign(delta_slice * X_prev_3d)     # (B, d_in, chunk)

        # accumulate over batch into this output chunk
        B_mat[:, o0:o1] = (err_mask * vote).sum(axis=0)

        # C_slice, err_mask, vote go out of scope here and are freed
        # before the next chunk is processed

    return B_mat


# ─────────────────────────────────────────────
#  Algorithm 1 — Top-K GFT, low-memory version
# ─────────────────────────────────────────────

def gft_train_lowmem(
    X_train, y_train,
    layer_sizes,
    total_iterations,
    batch_size,
    k_start,
    k_end,
    p_min,
    w_bit,
    runtime_weights_dir,
    out_chunk=16,         # NEW: output-feature chunk size for the B-matrix
    seed=42,
):
    """
    Memory-efficient GFT training. Identical results to the original
    gft_train, but the contribution tensor is streamed in output-feature
    chunks of size `out_chunk`. Lower out_chunk = less peak memory, slightly
    slower. On a PYNQ-Z2 try out_chunk in [8, 32].
    """
    rng = np.random.default_rng(seed)
    max_int = int(2 ** (w_bit - 1) - 1)
    L = len(layer_sizes) - 1

    if batch_size > len(X_train):
        batch_size = len(X_train)

    # sparse ternary init (same as original)
    weights = [np.load(os.path.join(runtime_weights_dir, f"{x}_0_StreamingDataflowPartition_{x}_MatrixVectorActivation_0.npy")) for x in range(1, L + 1)]

    activations = [relu] * (L - 1) + [lambda x: x]
    loss_history = []
    N = len(X_train)

    for t in range(total_iterations):
        k_frac = k_start + (k_end - k_start) * (t / total_iterations)

        idx = rng.choice(N, size=batch_size, replace=False)
        X_batch = X_train[idx]
        y_batch = y_train[idx]

        # forward — no C stored
        layer_outputs = forward_lowmem(X_batch, weights, activations)

        logits = layer_outputs[-1]
        loss, delta = cross_entropy_loss(logits, y_batch)
        loss_history.append(loss)

        # backward
        for l in range(L - 1, -1, -1):
            W = weights[l]
            d_in, d_out = W.shape
            k_count = max(1, int(k_frac * d_in * d_out))

            X_prev = layer_outputs[l - 1] if l > 0 else X_batch

            # chunked B-matrix — peak temp is (B, d_in, out_chunk)
            B_mat = compute_B_matrix(delta, X_prev, W, out_chunk)

            # top-k selection
            abs_B_flat = np.abs(B_mat).ravel()
            if k_count < len(abs_B_flat):
                threshold = np.partition(abs_B_flat, -k_count)[-k_count]
            else:
                threshold = 0
            B_topk = np.where(np.abs(B_mat) >= threshold, B_mat, 0)

            # probabilities + stochastic flip
            P = dynamic_probability(B_topk, p_min)
            rand_vals = rng.random(W.shape)
            flip_mask = (rand_vals < P) & (B_topk != 0)
            weights[l] = np.where(
                flip_mask,
                np.clip(W - np.sign(B_topk), -max_int, max_int),
                W
            )

            # propagate delta
            delta = delta @ W.T

        print(f"  Iter {t+1:4d}/{total_iterations} | loss = {loss:.4f} | k = {k_frac:.3f}")

    return weights, loss_history


def predict(X, weights):
    activations = [relu] * (len(weights) - 1) + [lambda x: x]
    layer_outputs = forward_lowmem(X, weights, activations)
    return np.argmax(layer_outputs[-1], axis=-1)


# ─────────────────────────────────────────────
#  Equivalence check + demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # toy 2-class dataset
    N = 400
    X0 = rng.normal(loc=[-1, -1], scale=0.5, size=(N // 2, 2))
    X1 = rng.normal(loc=[ 1,  1], scale=0.5, size=(N // 2, 2))
    X = np.vstack([X0, X1]).astype(np.float32)
    y = np.array([0] * (N // 2) + [1] * (N // 2))
    perm = rng.permutation(N)
    X, y = X[perm], y[perm]
    X_train, y_train = X[:320], y[:320]
    X_test,  y_test  = X[320:], y[320:]

    # ── Verify chunked B-matrix matches the unchunked computation ──
    print("Equivalence check: chunked vs full B-matrix")
    Bsz, d_in, d_out = 16, 12, 20
    delta_t = rng.normal(size=(Bsz, d_out))
    Xp_t    = rng.normal(size=(Bsz, d_in))
    W_t     = rng.choice([-1, 0, 1], size=(d_in, d_out)).astype(float)

    # full (original) computation
    C_full   = Xp_t[:, :, None] * W_t[None, :, :]
    mask_full = (delta_t[:, None, :] * C_full) < 0
    vote_full = np.sign(delta_t[:, None, :] * Xp_t[:, :, None])
    B_full    = (mask_full * vote_full).sum(axis=0)

    for chunk in (1, 4, 7, 20):
        B_chunked = compute_B_matrix(delta_t, Xp_t, W_t, chunk)
        ok = np.array_equal(B_full, B_chunked)
        print(f"  out_chunk={chunk:2d}: identical = {ok}")

    print("\n" + "=" * 50)
    print("Low-memory GFT demo (out_chunk=8)")
    print("=" * 50)
    trained, losses = gft_train_lowmem(
        X_train, y_train,
        layer_sizes=[2, 32, 16, 2],
        total_iterations=600, batch_size=64,
        k_start=0.75, k_end=0.1, p_min=0.01,
        max_int=1, out_chunk=8, seed=7,
    )
    acc = np.mean(predict(X_test, trained) == y_test) * 100
    print(f"\nTest accuracy: {acc:.1f}%  |  final loss: {losses[-1]:.4f}")