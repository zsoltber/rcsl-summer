"""
Software simulation loop for DiscreteDE weight fine-tuning
===========================================================
Mirrors the hardware loop exactly, but runs inference through your Brevitas
model in software instead of the FPGA accelerator.

The point of this script is to:
  1. Verify your ask/tell loop is wired correctly before touching hardware
  2. Check weights are being modified and loss is responding
  3. Get a rough budget estimate (how many evals before convergence)
  4. Catch shape/dtype issues early

The fitness function is identical to the hardware version: 1 - accuracy
on your local validation set. The only thing that changes between this
script and the hardware version is the evaluate() function.

Dependencies:
    pip install nevergrad torch brevitas numpy

Assumptions:
    - Your Brevitas model is already trained and exported
    - You have a local validation DataLoader (your distribution-shifted data)
    - Your weight .npy files are per-layer, matching the model's state_dict keys
    - Weights are INT4 symmetric: values in [-7, 7]
"""

import copy
import numpy as np
import torch
import nevergrad as ng


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — Model loading
#
#  Load your Brevitas model and freeze it in eval mode.
#  Brevitas quantised models behave like standard PyTorch models for inference
#  — you just call model(x) and get float logits back, quantisation happens
#  internally during the forward pass.
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_class, checkpoint_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load your Brevitas model from a checkpoint.
    Replace model_class with your actual model definition import.
    """
    model = model_class()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    # Disable all gradients — we never backprop in this workflow
    for p in model.parameters():
        p.requires_grad_(False)

    return model


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — Weight injection
#
#  This is the software equivalent of writing a .dat file to the FPGA.
#  We temporarily overwrite the target layer's weights in the model's
#  state_dict with the candidate array proposed by nevergrad, run inference,
#  then restore the original weights for the next iteration.
#
#  Important: Brevitas stores quantised weights as float tensors internally
#  (the int values are represented as floats scaled by the step size).
#  When injecting, cast your INT4 numpy array to float before assignment.
# ─────────────────────────────────────────────────────────────────────────────

def inject_weights(model, layer_key: str, weights: np.ndarray):
    """
    Overwrite a single layer's weights in the model with a candidate array.
    weights: INT4 numpy array, shape matching the original layer weight tensor.
    """
    weight_tensor = torch.tensor(
        weights.astype(np.float32),
        dtype=torch.float32
    )
    # Direct assignment into the state dict parameter
    # Use no_grad to avoid any autograd tracking
    with torch.no_grad():
        param = dict(model.named_parameters())[layer_key]
        param.copy_(weight_tensor)


def restore_weights(model, layer_key: str, original_weights: np.ndarray):
    """Restore a layer's weights to the original baseline values."""
    inject_weights(model, layer_key, original_weights)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — Fitness evaluation
#
#  Run inference on your local validation set and return 1 - accuracy.
#  This is identical in signature to what the hardware version will use —
#  the only internal difference is forward pass through model vs FPGA.
#
#  Your val_loader should yield (inputs, labels) batches of your local
#  distribution-shifted data, NOT the global training distribution.
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_accuracy(model, val_loader, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> float:
    """
    Run full validation set through model, return accuracy in [0, 1].
    """
    correct = 0
    total   = 0

    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs    = model(inputs)
            predicted  = outputs.argmax(dim=1)
            correct   += (predicted == labels).sum().item()
            total     += labels.size(0)

    return correct / total if total > 0 else 0.0


def compute_loss(model, val_loader, device: str = "cpu") -> float:
    """Loss = 1 - accuracy. Nevergrad minimises, so lower is better."""
    return 1.0 - evaluate_accuracy(model, val_loader, device)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — Single layer software simulation loop
#
#  This mirrors the hardware loop from nevergrad_discrete_de.py exactly.
#  The structure is:
#    ask() → inject weights → forward pass → compute loss → tell()
#
#  layer_key:        the state_dict key for the layer you're optimising,
#                    e.g. "features.0.weight" or "conv1.weight"
#  candidate_indices: pre-filtered flat indices from your importance scoring.
#                    None = search all weights in the layer (not recommended
#                    for large layers — use filtering first).
# ─────────────────────────────────────────────────────────────────────────────

def run_software_sim(
    model,
    val_loader,
    layer_key: str,
    baseline_npy: np.ndarray,       # loaded from your .npy file
    candidate_indices: np.ndarray,  # pre-filtered indices, shape (K,)
    budget: int = 300,
    patience: int = 50,             # early stop if no improvement
    loss_threshold: float = 0.01,   # early stop if loss drops below this
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
):
    """
    Software simulation of the DiscreteDE weight search for one layer.

    Returns:
        best_weights: full layer weight array (same shape as baseline_npy)
        best_loss:    best loss achieved
        history:      list of loss values per evaluation
    """
    bw = 4
    I  = 2**(bw - 1) - 1   # = 7

    flat_baseline = baseline_npy.flatten()
    subset        = flat_baseline[candidate_indices]

    # ── Parametrisation ──────────────────────────────────────────────────────
    param = ng.p.Array(
        init=subset.astype(float),
        lower=-I,
        upper=I,
    ).set_integer_casting()

    optimizer = ng.optimizers.DiscreteDE(
        parametrization=param,
        budget=budget,
        num_workers=1,
    )

    # ── Baseline measurement ─────────────────────────────────────────────────
    # Always measure baseline accuracy first so you know what you're starting
    # from and can detect regressions.
    baseline_loss = compute_loss(model, val_loader, device)
    if verbose:
        print(f"Baseline loss: {baseline_loss:.4f}  (acc: {1-baseline_loss:.4f})")
        print(f"Searching over {len(candidate_indices)} / {len(flat_baseline)} weights")
        print(f"Budget: {budget} evaluations\n")

    best_loss       = baseline_loss
    best_weights    = baseline_npy.copy()
    evals_since_imp = 0
    history         = [baseline_loss]

    # ── Main ask/tell loop ───────────────────────────────────────────────────
    for i in range(budget):

        candidate = optimizer.ask()

        # Reconstruct full weight array with proposed subset values
        full_weights = flat_baseline.copy()
        full_weights[candidate_indices] = candidate.value
        full_weights_shaped = full_weights.reshape(baseline_npy.shape).astype(np.int8)

        # Inject into model, evaluate, restore
        inject_weights(model, layer_key, full_weights_shaped)
        loss = compute_loss(model, val_loader, device)
        restore_weights(model, layer_key, baseline_npy)

        optimizer.tell(candidate, loss)
        history.append(loss)

        if loss < best_loss:
            best_loss       = loss
            best_weights    = full_weights_shaped.copy()
            evals_since_imp = 0
            if verbose:
                print(f"  iter {i:4d} | loss {best_loss:.4f} | acc {1-best_loss:.4f}  ✓")
        else:
            evals_since_imp += 1

        # Early stopping
        if best_loss <= loss_threshold:
            if verbose:
                print(f"\nConverged at iter {i} — loss {best_loss:.4f} ≤ threshold")
            break
        if evals_since_imp >= patience:
            if verbose:
                print(f"\nStagnated at iter {i} — no improvement for {patience} evals")
            break

    if verbose:
        improvement = baseline_loss - best_loss
        print(f"\nDone. Loss {baseline_loss:.4f} → {best_loss:.4f}  "
              f"(Δ = {improvement:+.4f}, {len(history)} evals used)")

    return best_weights, best_loss, history


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 — Layer-wise software simulation
#
#  Iterates through layers in priority order, passing improved weights forward.
#  Each layer's best result becomes the new model state for the next layer.
#  After each layer, save the improved .npy so you have a checkpoint.
# ─────────────────────────────────────────────────────────────────────────────

def run_layerwise_software_sim(
    model,
    val_loader,
    layer_configs: list,    # list of dicts, see example in __main__ below
    budget_per_layer: int = 300,
    patience: int = 50,
    output_dir: str = ".",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    layer_configs: list of dicts, each with keys:
        "layer_key"   : str   — state_dict key, e.g. "conv1.weight"
        "npy_path"    : str   — path to baseline .npy for this layer
        "candidates"  : np.ndarray — pre-filtered candidate indices
    """
    import os
    results = {}

    for cfg in layer_configs:
        layer_key = cfg["layer_key"]
        npy_path  = cfg["npy_path"]
        candidates = cfg["candidates"]

        print(f"\n{'='*60}")
        print(f"Layer: {layer_key}")
        print(f"{'='*60}")

        baseline = np.load(npy_path)

        best_w, best_loss, history = run_software_sim(
            model=model,
            val_loader=val_loader,
            layer_key=layer_key,
            baseline_npy=baseline,
            candidate_indices=candidates,
            budget=budget_per_layer,
            patience=patience,
            device=device,
        )

        # Save improved weights as new baseline for this layer
        out_path = os.path.join(output_dir, f"{layer_key.replace('.', '_')}_improved.npy")
        np.save(out_path, best_w)
        print(f"Saved improved weights → {out_path}")

        # Update the model state so subsequent layers see the improvement
        inject_weights(model, layer_key, best_w)

        results[layer_key] = {
            "best_loss": best_loss,
            "history":   history,
            "npy_path":  out_path,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 — Diagnostic utilities
#
#  Useful checks to run before committing to a full search.
# ─────────────────────────────────────────────────────────────────────────────

def check_weights_are_changing(model, layer_key: str, baseline_npy: np.ndarray,
                                candidate_indices: np.ndarray, n_samples: int = 5):
    """
    Sanity check: ask nevergrad for a few candidates and print how many
    weights actually differ from baseline. If this is always 0, something
    is wrong with your parametrisation or indices.
    """
    bw = 4
    I  = 2**(bw - 1) - 1

    flat_baseline = baseline_npy.flatten()
    subset        = flat_baseline[candidate_indices]

    param = ng.p.Array(
        init=subset.astype(float), lower=-I, upper=I
    ).set_integer_casting()

    optimizer = ng.optimizers.DiscreteDE(
        parametrization=param, budget=100, num_workers=1
    )

    print(f"Sanity check — weight change distribution over {n_samples} samples:")
    for i in range(n_samples):
        c = optimizer.ask()
        proposed = c.value
        n_changed = np.sum(proposed != subset)
        delta = proposed - subset
        print(f"  sample {i}: {n_changed} weights changed | "
              f"max Δ = {np.abs(delta).max()}  mean |Δ| = {np.abs(delta).mean():.2f}")
        optimizer.tell(c, 1.0)   # dummy loss for sanity check


def check_weight_distribution(baseline_npy: np.ndarray, layer_key: str):
    """
    Print distribution of weight values in a layer.
    Useful for confirming all values are in [-7, 7] and checking
    if the distribution is symmetric / healthy after training.
    """
    flat = baseline_npy.flatten()
    values, counts = np.unique(flat, return_counts=True)
    total = len(flat)
    print(f"\nWeight distribution for {layer_key}  (n={total}):")
    for v, c in zip(values, counts):
        bar = "█" * int(30 * c / counts.max())
        print(f"  {v:+3d}  {bar}  {c:5d}  ({100*c/total:.1f}%)")
    print(f"  min={flat.min()}  max={flat.max()}  "
          f"mean={flat.mean():.2f}  std={flat.std():.2f}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN — example usage
#
#  Replace the placeholders below with your actual model, dataloader,
#  layer keys, .npy paths, and candidate indices.
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Replace these with your actual objects ───────────────────────────────
    # from your_model import YourBrevitasModel
    # model      = load_model(YourBrevitasModel, "checkpoint.pth")
    # val_loader = your_local_dataloader(batch_size=64)
    #
    # For now, stubs so the script is runnable without your model:
    model      = None   # replace
    val_loader = None   # replace

    # ── Example: single layer sim ────────────────────────────────────────────
    if model is not None and val_loader is not None:

        baseline = np.load("layer0_weights.npy")

        # Your candidate indices from importance filtering
        # (replace with your actual filtered set)
        candidates = np.random.choice(baseline.size, size=50, replace=False)

        # Step 1: sanity check weights are actually changing
        check_weight_distribution(baseline, "conv1.weight")
        check_weights_are_changing(model, "conv1.weight", baseline, candidates)

        # Step 2: run the search
        best_weights, best_loss, history = run_software_sim(
            model=model,
            val_loader=val_loader,
            layer_key="conv1.weight",
            baseline_npy=baseline,
            candidate_indices=candidates,
            budget=300,
            patience=50,
        )

        np.save("conv1_weights_improved.npy", best_weights)

    # ── Example: layer-wise sim ───────────────────────────────────────────────
    if model is not None and val_loader is not None:

        layer_configs = [
            {
                "layer_key": "features.6.weight",   # last conv — highest leverage
                "npy_path":  "layer6_weights.npy",
                "candidates": np.random.choice(1000, 80, replace=False),
            },
            {
                "layer_key": "features.4.weight",
                "npy_path":  "layer4_weights.npy",
                "candidates": np.random.choice(2000, 100, replace=False),
            },
            {
                "layer_key": "features.2.weight",
                "npy_path":  "layer2_weights.npy",
                "candidates": np.random.choice(4000, 120, replace=False),
            },
        ]

        results = run_layerwise_software_sim(
            model=model,
            val_loader=val_loader,
            layer_configs=layer_configs,
            budget_per_layer=300,
            patience=50,
            output_dir="improved_weights",
        )