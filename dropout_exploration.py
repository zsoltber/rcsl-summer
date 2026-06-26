import numpy as np
import sys
sys.path.insert(1, "/home/xilinx/jupyter_notebooks/ContinualLearningMB")
from utils import hil_validation # runs on PL validation using given set of weights
from finn_pack_npy_only import pack_layer
import os
import json
from collections import defaultdict

rng = np.random.default_rng(seed=42) # reproducability of runs

# bw = 4
# target_fps_k = 100
# starting_weight_dir = f""
# starting_weights = [np.load(os.path.join(starting_weight_dir, f"{x+1}_0_StreamingDataflowPartition_{x+1}_MatrixVectorActivation_0.dat") for x in range(len(weights))]

def dynamic_probability (topk_mat, p_min=0.001):
    d_max = topk_mat.max()
    d_min = topk_mat[topk_mat > 0].min() if (topk_mat > 0).any() else 0
    
    # no destructive weight marking
    if d_max == 0:
        return np.zeros_like(topk_mat, dtype=float)
    elif d_max == d_min:
        prob = np.where(topk_mat > 0, p_min, 0.0)
    else:
        prob = np.where(
            topk_mat > 0,
            (topk_mat - d_min) / (d_max - d_min),
            0.0
        )

        # assign at least p_min flip probability to all weights included in the topk
        prob = np.where(topk_mat > 0, np.clip(prob, p_min, 1), 0.0)

    return prob

def prune_mod (layer, mask):
    return np.where(mask, 0, layer)

def half_mod (layer, mask):
    return np.where(mask, layer // 2, layer)

def sign_flip_mod (layer, mask):
    return np.where(mask, (-1) * layer, layer)

def shift_to_zero_mod (layer, mask):
    return np.where(mask, layer - np.sign(layer), layer)

def dropout_select (weights, population_size, pruning_rate, modification_type, k=None, p_min=0.005, selection_method="accumulate"):
    if k is None:
        k = pruning_rate / 2

    MODIFICATION_TYPES = ["Prune", "Half", "SignFlip", "ShiftToZero", "All"]
    SELECTION_METHODS = ["accumulate", "best_mask"]
    assert modification_type in MODIFICATION_TYPES, "Selected modification type not found."
    assert selection_method in SELECTION_METHODS, "Selected selection method not found."
    val_dir = "deploy/driver/hil_validation/b4_100k"

    L = len(weights)
    baseline_accuracy = hil_validation(weights)

    damage = [np.zeros_like(w) for w in weights]
    modified_weights = defaultdict(list) if modification_type == "All" else []

    weights_json = "deploy/driver/runtime_weights_initial/b4_100k/weights.json"
    with open(weights_json, "r") as f:
        layer_info = json.load(f)

    for layer_id in range(L):
        k_count = max(1, int(k * weights[layer_id].size))
        pe = layer_info[f"MatrixVectorActivation_{layer_id}"]["PE"]
        simd = layer_info[f"MatrixVectorActivation_{layer_id}"]["SIMD"]
        wdt = layer_info[f"MatrixVectorActivation_{layer_id}"]["WDT"]

        layer_dat_output = os.path.join(val_dir, f"{layer_id + 1}_0_StreamingDataflowPartition_{layer_id + 1}_MatrixVectorActivation_0.dat")

        best_delta = 0.0
        best_mask = np.zeros(weights[layer_id].shape, dtype=bool)

        # test population_size random dropout patterns to identify damaging weights in current layer
        for p in range(population_size):
            pruning_filter = rng.choice([True, False], weights[layer_id].shape, p=[pruning_rate, 1 - pruning_rate]) # create a mask for pruning_rate*100% of weights

            # new_w_layer is a copy of a given layer
            pruned_layer = weights[layer_id].copy()
            pruned_layer[pruning_filter] = 0 # prune copy for testing

            # copy a set of all weights and place the pruned layer back among the originals
            new_weights = list(weights)
            new_weights[layer_id] = pruned_layer

            # replace layer in HIL folder with pruned version - only recompute the current layer
            pack_layer(pruned_layer, layer_dat_output, pe, simd, wdt)

            # run HIL validation on the dataset on the FPGA to determine accuracy change due to pruning
            accuracy = hil_validation()

            # delta is new accuracy vs baseline
            delta_acc = accuracy - baseline_accuracy
            print(f"Delta for layer {layer_id + 1}, population member {p + 1}: {delta_acc:.2f}%")

            if selection_method == "accumulate":
                # the set of pruned test weights are marked as destructive i.e. hurting output if
                # their removal improves model performance
                damage[layer_id][pruning_filter] += delta_acc
            else:  # best_mask: keep whichever single mask gave the largest positive accuracy gain
                if delta_acc > best_delta:
                    best_delta = delta_acc
                    best_mask = pruning_filter.copy()

        if selection_method == "accumulate":
            damage_flat = damage[layer_id].ravel()
            damage_flat = np.clip(damage_flat, 0, None) # replace neg. values with 0 as a neg. score means weight is more likely needed
            # if k is larger than the layer size (shouldnt happen), then k = layer size, set topk threshold
            if k_count < len(damage_flat):
                threshold = np.partition(damage_flat, -k_count)[-k_count]
            else:
                threshold = 0

            # set all non topk elements to zero
            original_shape = weights[layer_id].shape
            damage_topk = np.where(damage_flat >= threshold, damage_flat, 0).reshape(original_shape)
            prob = dynamic_probability(damage_topk, p_min).reshape(original_shape)

            # stochastic weight shifting
            rand_vals = rng.random(original_shape)
            mod_mask = (rand_vals < prob) & (damage_topk != 0) # stochastically selected weight pos to modify
        else:  # best_mask
            mod_mask = best_mask  # use the mask from the single best-performing population member

        # modify weights according to shifting style
        mod_layer = weights[layer_id]
        if modification_type == "All":
            modified_weights["Prune"].append(prune_mod(mod_layer, mod_mask))
            modified_weights["Half"].append(half_mod(mod_layer, mod_mask))
            modified_weights["SignFlip"].append(sign_flip_mod(mod_layer, mod_mask))
            modified_weights["ShiftToZero"].append(shift_to_zero_mod(mod_layer, mod_mask))
        elif modification_type == "Prune":
            mod_layer = prune_mod(mod_layer, mod_mask)
            modified_weights.append(mod_layer)
        elif modification_type == "Half":
            mod_layer = half_mod(mod_layer, mod_mask)
            modified_weights.append(mod_layer)
        elif modification_type == "SignFlip":
            mod_layer = sign_flip_mod(mod_layer, mod_mask)
            modified_weights.append(mod_layer)
        else: # ShiftToZero 
            mod_layer = shift_to_zero_mod(mod_layer, mod_mask)
            modified_weights.append(mod_layer)
        
        pack_layer(weights[layer_id], layer_dat_output, pe, simd, wdt) # reset layer to original

    return modified_weights
        
