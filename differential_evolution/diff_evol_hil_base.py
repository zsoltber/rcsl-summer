import numpy as np
import nevergrad as ng # library by facebook research implementing evolutionary and gradient free optimiser methods
from finn_pack_npy_only import pack_layer
# loading onto fpga can be done using FINNExampleOverlay.load_runtime_weights(), the built in method
# from there on, calculating the accuracy / loss is trivial, pass in a batch and compare I/O
import pl_acc_eval

# ── Your existing methods (assumed signatures) ───────────────────────────────
# pack_to_dat(weights: np.ndarray, path: str) -> None
# load_onto_fpga(dat_path: str) -> None
# evaluate_accuracy_on_fpga() -> float   returns accuracy in [0, 1]

def run_hardware_tuning_loop(
    npy_path: str,
    candidate_indices: np.ndarray, # mask used to modify
    budget: int = 300,
    patience: int = 50,
):
    bw = 4
    I  = 2**(bw - 1) - 1   # = 7

    baseline  = np.load(npy_path) # iterate through the layers of the network
    flat_base = baseline.flatten()
    subset    = flat_base[candidate_indices] # might be okay to initialise randomly for now and then see if there are any masks that improve performance

    # ── Parametrisation ──────────────────────────────────────────────────────
    param = ng.p.Array(
        init=subset.astype(float),
        lower=-I,
        upper=I,
    ).set_integer_casting() # less overhead compared to Choice or TransitionChoice

    # ── Optimiser Selection ──────────────────────────────────────────────────
    # Differential Evolution
    optimizer = ng.optimizers.DiscreteDE(
        parametrization=param,
        budget=budget,
        num_workers=0,
    )

    # ── Baseline measurement ─────────────────────────────────────────────────
    baseline_loss = 1.0 - pl_acc_eval() # either pack weights before or involve pack_layer into eval (?)
    print(f"Baseline loss: {baseline_loss:.4f}  (acc: {1-baseline_loss:.4f})")

    best_loss       = baseline_loss
    best_weights    = baseline.copy()
    loss_history = [baseline_loss]
    evals_since_imp = 0

    # ── Ask/tell loop ────────────────────────────────────────────────────────
    # the nevergrad optimiser keeps track of the population, modifies based on candidate loss
    # new sample solution can be requested using optimiser.ask, loss can be fed in using optimiser.tell
    for i in range(budget):
        candidate = optimizer.ask()

        # Reconstruct full weight array
        full_weights = flat_base.copy()
        full_weights[candidate_indices] = candidate.value
        full_weights_shaped = full_weights.reshape(baseline.shape).astype(np.int8)

        # ↓ your existing pipeline
        pack_layer(full_weights_shaped, "candidate.dat")
        # load_onto_fpga("candidate.dat") -> accel.load_runtime_weights() on an existing FINNExampleOverlay object
        loss = 1.0 - pl_acc_eval()
        loss_history.append(loss)

        optimizer.tell(candidate, loss)

        if loss < best_loss:
            best_loss       = loss
            best_weights    = full_weights_shaped.copy()
            evals_since_imp = 0
            print(f"  iter {i:4d} | loss {best_loss:.4f} | acc {1-best_loss:.4f}  ✓")
        else:
            evals_since_imp += 1

        if evals_since_imp >= patience:
            print(f"Stagnated at iter {i} — stopping")
            break

    # Save best weights back as npy
    out_path = npy_path.replace(".npy", "_improved.npy") # save to the runtime weight dir and convert result to dat ... 
    # continue evol tuning with the already modified layers
    np.save(out_path, best_weights)
    print(f"Saved → {out_path}")

    return best_weights, best_loss, loss_history
