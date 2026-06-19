import numpy as np
import sys
sys.path.insert(1, "/home/xilinx/jupyter_notebooks/ContinualLearningMB")
from utils import hil_validation # runs on PL validation using given set of weights
from finn_pack_npy_only import pack_layer # method converting a single .npy file into a FINN .dat file
import os

rng = np.random.default_rng(seed=42) # reproducability of runs

# bw = 4
# target_fps_k = 100
# starting_weight_dir = f""
# starting_weights = [np.load(os.path.join(starting_weight_dir, f"{x+1}_0_StreamingDataflowPartition_{x+1}_MatrixVectorActivation_0.dat") for x in range(len(weights))]
L = len(starting_weights)
baseline_accuracy = hil_validation(starting_weights)

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

def dropout_select (population_size, weights, baseline_accuracy, pruning_rate, modification_type, k, p_min):
    MODIFICATION_TYPES = ["Prune", "Half", "SignFlip", "ShiftToZero"]
    assert modification_type in MODIFICATION_TYPES, "Selected modification type not found."
    
    damage = [np.zeros_like(w) for w in weights]
    modified_weights = []

    # accumulate damages for each layer by testing population sizes
    # the damage list aggregates weight effects on final accuracy to identify if there are any weights
    # that the network in its current state is better without or ones that are crucial for a correct prediction
    for layer_id in range(L):
        k_count = max(1, int(k * weights[layer_id].size))
        
        # test population_size random dropout patterns to identify damaging weights in current layer
        for p in range(population_size):
            pruning_filter = rng.choice([True, False], weights[layer_id].shape, p=[pruning_rate, 1 - pruning_rate]) # create a mask for pruning_rate*100% of weights

            # new_w_layer is a copy of a given layer
            pruned_layer = weights[layer_id].copy()
            pruned_layer[pruning_filter] = 0 # prune copy for testing

            # copy a set of all weights and place the pruned layer back among the originals
            new_weights = list(weights)
            new_weights[layer_id] = pruned_layer

            # run HIL validation on the dataset on the FPGA to determine accuracy change due to pruning
            accuracy = hil_validation(new_weights)

            # delta is new accuracy vs baseline
            delta_acc = accuracy - baseline_accuracy

            # the set of pruned test weights are marked as destructive i.e. hurting output if 
            # their removal improves model performance
            damage[layer_id][pruning_filter] += delta_acc 

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

        # modify weights according to shifting style
        mod_layer = weights[layer_id]
        if modification_type == "Prune":
            mod_layer = np.where(
                mod_mask,
                0,
                mod_layer
            )
        elif modification_type == "Half":
            mod_layer = np.where(
                mod_mask,
                mod_layer // 2  ,
                mod_layer
            )
        elif modification_type == "SignFlip":
            mod_layer = np.where(
                mod_mask,
                (-1) * mod_layer,
                mod_layer
            )
        else: # ShiftToZero 
            mod_layer = np.where(
                mod_mask,
                mod_layer - np.sign(mod_layer),
                mod_layer
            )
        
        modified_weights.append(mod_layer)

    acc = hil_validation(modified_weights) # runs an eval with new weights on the test dataset
    print(f"Accuracy of modified weights: {acc:.2f}%")

    return modified_weights
        
