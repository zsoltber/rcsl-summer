import numpy as np
import os
import sys
sys.path.insert(1, "/home/xilinx/jupyter_notebooks/ContinualLearningMB")
from utils import hil_validation

# ---------- HELPER FUNCTIONS ----------
# Activations, Loss Function

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def cross_entropy_loss(logits, labels):
    # PARAMS
    # logits : (B, num_classes)
    # classes : (B,)
    
    # RETURNS
    # Returns scalar loss and delta at output layer (B, num_classes)
    
    probs = softmax(logits)
    B = len(labels)
    loss = -np.mean(np.log(probs[np.arange(B), labels] + 1e-9))
    
    # delta^(L) = predicted_probabilities - one_hot_label (cross-entropy + softmax grad)???
    delta = probs.copy()
    delta[np.arange(B), labels] -= 1 # select probs corresponding to correct labels and get their delta
    delta /= B
    return loss, delta

# Modified Forward method - also stores the contribution of each weight towards the layer output (quantifying impact)
# this is used later in comparison with delta values to select destructive weights, which propagate bad final probs

def modified_fwd(X_batch, weights, activations):
    # PARAMS
    # X_batch : (B, d0) input batch
    # weights : list of L arrays, each having (d_{l-1}, d_l) l = 1, ..., num_layers
    # activations : list of L callables (activation function for each layer - e.g. ReLU)
    
    # RETURNS
    # layer_outputs : list of L arrays, X^(l), each (B, d_l)
    # contributions : list of L arrays C^(l), each (B, d_{l-1}, d_l)
    
    layer_outputs = []
    contributions = []
    
    X_prev = X_batch # raw input
    
    for l, (W, sigma) in enumerate(zip(weights, activations)): # current layer, layer weights, layer act. func.
        # C^(l)_bio = x^(l)_bi * w^(l)_io
        # (B, d_{l-1}, d_l)
        # X_prev is (B, d_l), W is (d_{l-1}, d_l)
        C = X_prev[:, :, np.newaxis] * W[np.newaxis, :, :] # (B, d_{l-1}, d_l)
        
        # v^(l) is the sum over input dimension of C^l_bio (equivalent to X_prev @ W)
        v = C.sum(axis=1)
        
        X_curr = sigma(v) # activation
        
        contributions.append(C) # might be pretty large, for 45270 test dataset, the full C array is 45270x490x256
        layer_outputs.append(X_curr)
        X_prev = X_curr
        
    return layer_outputs, contributions

# Dynamic Probability - gives modification probabilities to each top k most destructive weights
def dynamic_probability(B_topk, p_min=0.001):
    # PARAMS
    # B_topk : (d_(l-1), d_l) subset of B matrix containing votes for most destructive k weights 
    #    (if w_io) not in top k, then b_io replaced by 0
    # p_min : float corresponding to lower bound of assigned shifting probabilities
    
    # RETURNS
    # P : float matrix of dim (d_(l-1), d_l) of probabilities between [p_min, 1.0] 
    #    (zero values for values not in topk most destructive)
    
    # work with abs magnitudes
    abs_B = np.abs(B_topk)
    
    b_max = abs_B.max()
    b_min = abs_B[abs_B > 0].min() if (abs_B > 0).any() else 0
    
    if b_max == 0:
        # no destructive weights to update
        return np.zeros_like(B_topk, dtype=float)
    
    if b_max == b_min:
        # all candidates hold equal importance, assign p_min to all of them
        P = np.where(abs_B > 0, p_min, 0.0)
        
    else:
        # normalise non zero entries to [0, 1]
        P = np.where(
            abs_B > 0,
            (abs_B - b_min) / (b_max - b_min),
            0.0
        )
        # floor at p_min so even b_min candidates have chance to be modified
        P = np.where(abs_B > 0, np.clip(P, p_min, 1.0), 0.0)
        
    return P   

# ---------- MAIN ALGORITHM IMPLEMENTATION ----------

# Top-K Gradient Free Training Loop - run a batch process, cross match weight contributions to deltas
# mark destructive weights, modify the worst ones in the quantisation direction

def gft_train (
    X_train, 
    y_train,
    layer_sizes,
    total_iterations,
    batch_size,
    k_start, 
    k_end, 
    p_min,
    w_bit,
    runtime_weights_dir,
    seed = 42
    ):
    
    """
    PARAMETERS
    
    X_train : input dataset
    y_train : output labels
    layer_sizes : list of layer sizes e.g. [490, 256, 256, 256, 12]
    training_iterations : number of loops to run
    batch_size : input subset size
    k_start/end : fraction of total weights to consider for modification
    p_min : minimum shift probability
    w_bit : model bit width - determines range of weight values
    runtime_weights_dir : directory where runtime weights are stored as .npy and .dat
    
    RETURNS
    
    weights : list of weight matrices
    loss_history : list of scalar loss values
    val_top1_history : top 1 accuracy over iterations on val dataset
    """
    best_weight_return = False
    rng = np.random.default_rng(seed)
    max_int = int(2 ** (w_bit - 1) - 1)
    L = len(layer_sizes) - 1 # length of layer_sizes list is one more than the number of weight files between these layers
    
    # load weights from runtime_weights_dir
    weights = [np.load(os.path.join(runtime_weights_dir, f"{x}_0_StreamingDataflowPartition_{x}_MatrixVectorActivation_0.npy")) for x in range(1, L + 1)]
    best_weights = weights
    
    activations = [relu] * (len(weights) - 1) + [lambda x: x] # ReLU for all layers except the last one
    
    best_acc = 77.55
    loss_history = []
    accuracy_history = []
    N = len(X_train) 
    
    if N < batch_size:
        raise ValueError(f"batch_size ({batch_size}) cannot exceed dataset size ({N})")
    
    for t in range(total_iterations):
        # Scheduler linearly decreasing mod. frac. as iterations progress
        k_frac = k_start + (k_end - k_start) * (t / total_iterations)
        
        # sample mini batch from dataset
        idx = rng.choice(N, size=batch_size, replace=False)
        X_batch = X_train[idx]
        y_batch = y_train[idx]
        
        # algorithm 2 - modified forward pass calculating layer outputs and contributions
        layer_outputs, contributions = modified_fwd(X_batch, weights, activations)
        
        # compute loss, initial delta at output
        logits = layer_outputs[-1] # last layer outputs
        loss, delta = cross_entropy_loss(logits, y_batch)
        loss_history.append(loss)
#         min_loss = loss if t == 0
        
        # backward pass 
        for l in range(L - 1, -1, -1):
            C = contributions[l]
            W = weights[l]
            d_in, d_out = W.shape
            
            # num weights to consider for update
            k_count = max(1, int(k_frac * d_in * d_out))
            
            """
            Compute B^(l): aggregation of error signal per weight
            
            For each (i, o), sum sign(delta_bo * x^(l-1)_bi) over batch (only if there is an error - c and del disagree)
            
            delta : (B, d_out)
            C : (B, d_in, d_out)
            
            Mismatch condition: delta_bo * C_bio < 0
            Sign contribution: sign(delta_bo * x^(l-1)_bi)
                x^(l-1) is the layer input, recoverable from:
                    1. C/W 
                        OR
                    2. layer_outputs[l-1] if l > 0 else it is X_batch
            """
            
            X_prev = layer_outputs[l - 1] if l > 0 else X_batch
            
            # Broadcast: delta (B, d_out) -> (B, 1, d_out)
            delta_3d = delta[:, np.newaxis, :]
            X_prev_3d = X_prev[:, :, np.newaxis]
            
            # Create error mask where delta and contribution disagree
            error_mask = (delta_3d * C) < 0
            
            # Vote on direction of modification for each sample
            vote = np.sign(delta_3d * X_prev_3d)
            
            # Accumulate votes where there is a mismatch
            B_mat = (error_mask * vote).sum(axis=0)
            
            # Select the top k most destructive weights based on their |B| value
            abs_B_flat = np.abs(B_mat).ravel()
            if k_count < len(abs_B_flat):
                threshold = np.partition(abs_B_flat, -k_count)[-k_count]
            else:
                threshold = 0
                
            B_topk = np.where(np.abs(B_mat) >= threshold, B_mat, 0)
            
            # compute shifting probabilities using Algorithm 3, dynamic_flip probabilities
            P = dynamic_probability(B_topk, p_min)
            
            # stochastic weight shifting
            rand_vals = rng.random(W.shape)
            flip_mask = (rand_vals < P) & (B_topk != 0)
            
            # move weights one step in their desired shifting direction unless they are at the limit plus-minus I
            weights[l] = np.where(
                flip_mask,
                np.clip(W - np.sign(B_topk), -max_int, max_int),
                W
            )
            
            # propagate delta to the previous layer
            # delta^(l-1) = delta^(l) @ W^(l).T (matmul with W transpose)
            
            delta = delta @ W.T # (B, d_(l-1))
            
        # console logging loss and accuracy (can extend by doing a forward pass on the FPGA for HIL validation)    
        # if (t + 1) % 50 == 0: # every 50 iterations (might be overkill for fine-tuning to have that range)
        
        # calculates accuracy, saves weights .dat files to runtime_weight_dir - default spec in utils
        acc = hil_validation(weights)
        if acc > best_acc:
            best_acc = acc
            best_weights = weights
        accuracy_history.append(acc)
        
        print(f"Iteration {t+1:4d}/{total_iterations} | loss on training batch = {loss:.4f} | acc. on entire set = {acc:.2f}%")
        # k = {k_frac:.5f}")
        
    if best_weights_return:      
        return best_weights, loss_history, accuracy_history
    else:
        return weights, loss_history, accuracy_history


    