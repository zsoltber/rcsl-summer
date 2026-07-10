# Mask Potency Identification for On-Device Evolutionary Tuning of FINN QNNs
### Technical Reference Document — July 2026

---

## 1. Context and Problem Statement

**Deployment setup.** A quantised (INT4) MLP is compiled with FINN into streaming
MVAU (MatrixVectorActivation) layers on the PL of a Zynq-class FPGA. Runtime
weight update is possible without resynthesis via FINN's decoupled/runtime
weight interface: weights are re-packed into `.dat` files (respecting each
layer's PE/SIMD folding) and loaded through `accelerator.load_runtime_weights()`.
The PS (ARM core) runs a Nevergrad `DiscreteDE` ask/tell loop:

- Parametrisation: `ng.p.Array(init=subset, lower=-I, upper=+I).set_integer_casting()`
  with `I = 2^(bw-1) - 1` (= 7 for INT4).
- A **candidate mask** selects ~10% of one layer's weights (currently random,
  seeded); only masked weights are exposed to the optimiser.
- Fitness = `1 - accuracy` measured by running inference on the PL over a fixed
  eval batch. Per-iteration: reconstruct full weight array, repack the layer's
  `.dat`, reload runtime weights, evaluate.
- Budget ≈ 300 evals, patience-based early stop, per-layer tuning
  (default `layer_index = -1`, the last layer).

**The problem.** Mask quality dominates outcome, and random masks are hit-or-miss
even for sub-1M-parameter networks. DE's effective budget is spent searching
dimensions the loss does not respond to. Goal: a program that, given a network
(with or without a small number of tuning/accuracy evaluations), ranks weights
by how *worthwhile* they are to include in a tuning mask, and produces
interpretable heuristics for why some masks beat others — so full tuning runs
start from educated guesses instead of trial-and-error random subsets.

---

## 2. What Makes a Weight "Potent" — Three Separable Ingredients

A good mask member needs all three; each can be scored independently:

1. **Sensitivity** — a ±1 integer step on this weight measurably moves the loss.
   Many weights are effectively dead: they multiply near-zero activations, or
   their contribution never crosses a downstream threshold (FINN activations are
   threshold comparisons, so small pre-activation changes are frequently
   absorbed without any output change). Dead weights waste DE dimensions.
2. **Headroom** — room to move on the integer lattice. A weight at +7 (INT4)
   can only step inward; half its move set is gone, and the blocked direction
   is often the useful one (that is frequently *why* QAT pushed it to the clip
   boundary). Headroom = `min(w + I, I - w)`; saturation flag = `|w| == I`.
3. **Shift-relevance** — tuning targets a *local deviation* from the global
   training distribution. The weights worth moving are those whose inputs
   (upstream activations) behave differently under local vs. global data. A
   weight can be sensitive and unsaturated yet irrelevant to the particular
   shift.

Random masks fail because uniform sampling gives most slots to weights failing
at least one ingredient, and DE convergence degrades with search dimensionality
(well documented in the zeroth-order literature — see Sparse-MeZO, §8).

---

## 3. Stage A — Free Offline Scores (Zero Hardware Evaluations)

All computable on a workstation from `starting_weights` plus one forward pass
over calibration batches. Cost: seconds to minutes.

### A1. Headroom / saturation
- **Definition:** `headroom(w) = min(w - (-I), I - w)`; or binary saturation
  flag `|w| == I`.
- **Why it matters:** saturated weights have an asymmetric, halved move set on
  the integer lattice; probes and DE moves against the clip are wasted evals.
- **Implementation:** one vectorised NumPy op over the weight tensors. Store
  per-weight uint8.

### A2. Activation energy (dead-input filter)
- **Definition:** for weight `w_ij` connecting input j → output neuron i,
  `E_j = mean over batch of |a_j|` where `a_j` is the (quantised) activation
  feeding the layer. Every weight in column j inherits score `E_j`.
- **Why it matters:** the loss contribution of `w_ij` is gated by `a_j`. If
  `a_j ≈ 0` almost always (post-ReLU/threshold dead-ish neuron), no integer
  step on `w_ij` can matter. Cheapest possible dead-weight filter.
- **Implementation:** forward hooks in the Brevitas model (or intermediate
  tensors of the bit-accurate emulator) on ~1–2k *local* samples; accumulate
  mean |activation| per neuron. Note this is a **column-structured** score —
  aligns with SIMD lanes (see §7).

### A3. Activation-shift score (distribution-shift relevance)
- **Definition:** `Δ_j = | E_local[a_j] - E_global[a_j] |`, optionally plus a
  variance term `| Var_local[a_j] - Var_global[a_j] |`, computed per input
  neuron j; weights in column j inherit it. A refined variant uses per-class
  conditional means or an MMD-style statistic per neuron.
- **Why it matters:** this is the only Stage-A score that targets ingredient 3
  directly. Neurons whose statistics moved under the local distribution mark
  the sub-circuit that the domain shift actually flows through; weights fed by
  unchanged neurons encode still-valid global structure and are better left
  frozen (also a catastrophic-forgetting argument — cf. EWC, §8).
- **Implementation:** two forward-hook passes (one on held-out global data, one
  on local data), same hooks as A2. Requires a modest local calibration set
  (hundreds of samples suffice for first-moment statistics).

### A4. Proxy gradient / empirical Fisher (offline only)
- **Definition:** `F_ij = E over local batch of (dL/dw_ij)^2`, gradients taken
  through the Brevitas model with straight-through estimators (STE) on the
  quantisers. Related saliency variants: SNIP score `|g·w|`, OBD's diagonal
  Hessian saliency.
- **Why it matters:** gradients are unavailable *on device*, but nothing
  forbids using them offline to *rank* weights. The FISH-mask result (§8) shows
  Fisher-top-k masks are strong parameter subsets for sparse fine-tuning; this
  is the strongest known baseline the probe-based scores must beat.
- **Caveats:** STE gradients are biased for low bitwidths; Fisher measures
  sensitivity around the *current* point on *current* data, not usefulness for
  the shift, and it ignores the integer lattice (an infinitesimal-sensitivity
  weight may respond very differently to a full ±1 step). Treat as baseline,
  not oracle.
- **Implementation:** standard PyTorch loop, accumulate `grad**2` per weight
  over local batches, no optimiser step.

---

## 4. Stage B — Integer-Lattice ES Probe (Core Algorithm)

The atomic move of the deployed search is Δw = ±1, so measure the loss response
to unit integer steps directly, gradient-free, via antithetic group
perturbations (SPSA / evolution-strategies style, adapted to the integer
lattice). Per-weight probing is O(N) and hopeless; random-group probing gives
every weight a score estimate in O(T) total evaluations.

### Estimator
Perturb a random group with random signs δ ∈ {−1,+1}^k. To first order,
`L(w+δ) − L(w−δ) ≈ 2 Σ_i δ_i u_i`, where `u_i` is weight i's unit-step loss
response. Correlating each weight's sign with the observed antithetic
difference across many random groups deconvolves individual contributions —
the same mechanism as ES/SPSA gradient estimation, but the "gradient" here is
the *finite unit-step* response, which is exactly the quantity relevant to
integer-snapped DE (unlike infinitesimal Fisher/gradient scores).

### Pseudocode
```python
S  = np.zeros(N)      # signed sensitivity accumulator
V  = np.zeros(N)      # visit counts
Wp = np.zeros(N)      # count: +step made things worse AND -step made things worse
L0 = eval_loss(w0)    # baseline, FIXED eval batch

for t in range(T):
    idx   = rng.choice(N, size=k, replace=False)      # probe group
    delta = rng.choice([-1, +1], size=k)
    delta = clip_to_headroom(w0, idx, delta)          # saturated: force inward

    Lp = eval_loss(apply(w0, idx, +delta))            # antithetic pair
    Lm = eval_loss(apply(w0, idx, -delta))
    g  = (Lp - Lm) / 2.0

    S[idx] += -g * delta          # positive S  =>  +1 step likely reduces loss
    V[idx] += 1
    if Lp > L0 and Lm > L0:       # both directions hurt: locally optimal notch
        Wp[idx] += 1

potency    = np.abs(S) / np.maximum(V, 1)
direction  = np.sign(S)                       # free warm-start for DE
stuckness  = Wp / np.maximum(V, 1)            # sensitive-but-stuck indicator
```

### Three-way weight classification
- **Potent-and-ripe:** high |S|, low stuckness — one direction consistently
  improves. Prime mask candidates; `direction` warm-starts the search.
- **Sensitive-but-stuck:** loss responds, but both ±1 worsen — the weight sits
  in a local notch; single-weight moves cannot help, only coordinated
  multi-weight moves might. These are the "difficult" weights, now identified
  rather than guessed.
- **Dead / noise-dominated:** S ≈ 0 across many visits.

### Variance and correctness controls
- **Fixed eval batch + common random numbers:** identical samples for every
  eval within a probe campaign, otherwise data noise swamps signal. Optionally
  two fixed batches to detect eval-set overfitting of scores.
- **Group size k:** larger k = more weights visited per eval but noisier
  attribution. Start with k ≈ 1–5% of layer size; halve if the top-score set
  is unstable across two half-campaigns (a cheap stability diagnostic:
  Spearman correlation between scores from the two halves).
- **Antithetic pairing** cancels even-order terms and roughly halves variance
  vs one-sided probes; keep it.
- **Loss resolution:** with a small eval batch, accuracy is quantised in steps
  of 1/batch. Prefer cross-entropy / margin-based loss over raw accuracy for
  the probe signal if the emulator exposes logits; on hardware, use a larger
  eval batch or soft outputs if the final layer permits.

### Coarse-to-fine, PE/SIMD-aligned
1. **Round 1 — column granularity:** perturb whole input-channel (SIMD-lane)
   groups (all weights of a column get the same δ). Scores columns cheaply and
   with far lower variance (fewer unknowns per eval).
2. **Round 2 — weight granularity inside top columns only.**
Bonus: column-structured masks localise `.dat` repacking work per iteration,
reducing PS-side overhead per DE step.

### Budget math
- Emulation (bit-accurate QONNX / NumPy MVAU replica): T = 300–500 antithetic
  pairs = 600–1000 forward passes over the eval batch → minutes on a
  workstation. Do the whole probe campaign here.
- Hardware only: eval batch 128–512, T = 150–250 pairs ≈ 300–500 evals —
  comparable to one current DE run (budget 300), but the resulting scores are
  reusable across many tuning runs instead of being consumed by one.

---

## 5. Stage C — Mask Assembly, Tournament, Interpretation

### C1. Assembly rules
- Do **not** take a naive top-q by score: add a diversity cap (max weights per
  output neuron / per column) or the mask collapses onto one hot neuron and DE
  dimensions become strongly correlated.
- Blend recipe (starting point): 60% ES-probe top, 20% activation-shift top,
  20% uniform random. The random slice keeps the pipeline honest and directly
  measures how much scoring buys over chance.
- Respect headroom: exclude or down-weight saturated weights (test as a
  hypothesis, §6).

### C2. Tournament protocol (the testing loop)
- **Mask families:** {ES-probe, empirical Fisher, activation-shift, headroom,
  magnitude (large-|w| and small-|w| variants), pure random} at matched mask
  size (e.g., 10% of layer).
- **Runs:** SHORT DE runs — budget 30–50, not 300 — × 3 seeds per mask, on
  identical fixed eval batches (common random numbers, paired comparisons).
  Mask quality separates within tens of evals; short runs afford ~20× more
  mask comparisons per unit budget.
- **Metrics:** best-loss-at-budget, and area-under-the-loss-curve (AUC) over
  evals (captures speed, not just endpoint). Report paired differences vs the
  random-mask control with a sign test / Wilcoxon across seeds and eval sets.
- **Validation split:** always report improvement on a *held-out* local set,
  never the tuning eval batch, to catch eval-batch overfitting.

### C3. Interpretability (the "why")
Log per-mask features: mean/max probe score, saturation fraction, layer index,
mean activation energy, mean activation-shift score, Fisher overlap fraction,
per-neuron concentration (Gini/entropy of counts per output neuron), column
structure fraction. Regress features against tournament outcomes — plain
linear fit and/or a small gradient-boosted model with feature importances.
Output: rules of the form "masks win when they avoid saturated weights and
concentrate on high-activation-shift columns in the last two layers" — the
educated-guess heuristics for launching full runs, and the direct answer to
"can we build a mask toggle optimiser / what is its cost and scalability".

---

## 6. Pre-Registered Hypotheses for the Tournament

1. **Shift beats Fisher for local adaptation:** activation-shift masks
   outperform Fisher masks specifically on local-deviation tuning, even if
   Fisher wins on generic loss reduction.
2. **Last-layer dominance at small budgets:** shortest credit-assignment path;
   consistent with current default `layer_index = -1`.
3. **Saturation exclusion helps at INT4** and the effect shrinks with
   bitwidth.
4. **Column-structured masks match unstructured ones** at equal size — a
   hardware-friendly result if true (cheaper repacking, coarser probe).
5. **Small-|w| preference:** ZO literature (Sparse-MeZO) reports perturbation
   noise hurts more on large weights; test whether the analogous effect holds
   for ±1 integer steps at INT4 (where |w| ∈ {0..7} and the "relative" step on
   small weights is huge — the effect may invert; worth testing both
   directions).

---

## 7. Practical / Implementation Notes

- **Emulation first.** Mask identification does not need the hardware; only
  final tuning does. Bit-accurate options: QONNX execution
  (`finn.core.onnx_exec` / qonnx runtime) of the streamlined model, Brevitas
  inference, or a NumPy replica of the MVAU integer matmul + threshold stack
  built from `starting_weights` + extracted thresholds. Verify emulator
  fidelity once: identical predictions to PL inference on ~1k samples before
  trusting probe scores. Turns 1–2 s hardware evals into µs–ms.
- **Weight packing:** hardware evals go through
  `pack_layer(np_layer, dat_path, pe, simd, wdt="INT4")` per layer; per-column
  masks minimise how much of the `.dat` changes per iteration.
- **Scoring reuse:** probe scores are a property of (network, local data
  snapshot); recompute only when the local distribution drifts appreciably —
  monitor via the A3 statistics themselves (they double as drift detectors).
- **Baseline hygiene:** `starting_weights_dir` is never written by tuning
  loops; `load_initial_weights()` restores the deployed state between
  tournament runs.
- **Statistics:** everything paired: same seeds, same eval batches across mask
  families; report distributions across seeds, not single runs.
- **DE warm start:** initialise part of the DE population at
  `w0 + direction` (from the probe) rather than all at `w0`.

---

## 8. Reading List

**Zeroth-order / forward-only fine-tuning and sparse-mask ZO (closest field):**
- Malladi et al., *Fine-Tuning Language Models with Just Forward Passes*
  (MeZO), NeurIPS 2023 — https://arxiv.org/abs/2305.17333
- Liu et al., *Sparse MeZO: Less Parameters for Better Performance in
  Zeroth-Order LLM Fine-Tuning*, 2024 — https://arxiv.org/abs/2402.15751
  (ZO applied to a chosen parameter subset; selection scheme favours
  small-magnitude weights; directly analogous problem one field over)
- *CurvZO: Adaptive Curvature-Guided Sparse Zeroth-Order Optimization*, 2026 —
  https://arxiv.org/abs/2603.21725 (argues predefined/random sparsity patterns
  in sparse ZO lack an importance-driven selection mechanism — the same gap
  this project attacks for evolutionary integer search)
- Spall, *Multivariate Stochastic Approximation Using a Simultaneous
  Perturbation Gradient Approximation* (SPSA), IEEE TAC 1992 — overview:
  https://www.jhuapl.edu/spsa/
- Salimans et al., *Evolution Strategies as a Scalable Alternative to
  Reinforcement Learning*, 2017 — https://arxiv.org/abs/1703.03864
  (antithetic sampling, per-parameter credit assignment from group perturbations)

**Saliency / parameter-importance metrics (Stage A):**
- LeCun, Denker, Solla, *Optimal Brain Damage*, NeurIPS 1989 — diagonal-Hessian
  saliency; the ancestor of all weight-importance scores.
- Lee et al., *SNIP: Single-shot Network Pruning based on Connection
  Sensitivity*, ICLR 2019 — https://arxiv.org/abs/1810.02340 (|g·w| saliency)
- Theis et al., *Faster Gaze Prediction with Dense Networks and Fisher
  Pruning*, 2018 — https://arxiv.org/abs/1801.05787 (empirical Fisher as
  importance)
- Sung et al., *Training Neural Networks with Fixed Sparse Masks* (FISH mask),
  NeurIPS 2021 — https://arxiv.org/abs/2111.09839 (Fisher-top-k parameter
  subsets for sparse fine-tuning; the baseline to beat)
- Kirkpatrick et al., *Overcoming Catastrophic Forgetting in Neural Networks*
  (EWC), 2017 — https://arxiv.org/abs/1612.00796 (Fisher as "importance to old
  task"; the freeze-side rationale for masks in continual learning)

**Sparse-subset fine-tuning as a paradigm:**
- Zhao et al., *Masking as an Efficient Alternative to Finetuning for
  Pretrained Language Models*, EMNLP 2020 — https://arxiv.org/abs/2004.12406
- Ben Zaken et al., *BitFit: Simple Parameter-Efficient Fine-tuning*, 2021 —
  https://arxiv.org/abs/2106.10199 (tiny structured subsets can carry a
  fine-tune; motivates structured masks)
- Frankle & Carbin, *The Lottery Ticket Hypothesis*, ICLR 2019 —
  https://arxiv.org/abs/1803.03635 (existence of privileged sparse
  subnetworks; conceptual backdrop for "some masks are just better")

**Quantisation / platform:**
- Umuroglu et al., *FINN: A Framework for Fast, Scalable Binarized Neural
  Network Inference*, FPGA 2017 — https://arxiv.org/abs/1612.07119
- Blott et al., *FINN-R: An End-to-End Deep-Learning Framework for Fast
  Exploration of QNNs*, ACM TRETS 2018 — https://dl.acm.org/doi/10.1145/3242897
- Bengio et al., *Estimating or Propagating Gradients Through Stochastic
  Neurons* (STE), 2013 — https://arxiv.org/abs/1308.3432 (why offline proxy
  gradients through quantisers are biased)
- FINN publications index (incl. folding, runtime weights, benchmarking):
  https://xilinx.github.io/finn/publications.html

---

## 9. Research Timeline (6 weeks, adjustable)

**Week 1 — Emulation harness + Stage A.**
Build/verify the bit-accurate emulator against PL inference (~1k samples,
require identical predictions). Implement A1–A4 scores with forward hooks;
produce per-layer score maps and sanity visualisations (heatmaps of headroom,
activation energy, shift score). Deliverable: `scores.npz` per layer +
fidelity report.

**Week 2 — Stage B probe.**
Implement the antithetic integer-lattice probe on the emulator (column round,
then weight round). Run the stability diagnostic (half-campaign Spearman) to
fix T and k. Produce the three-way classification (ripe / stuck / dead) and
compare rank correlation of probe scores vs Stage-A scores (are they measuring
the same thing? disagreements are the interesting weights). Deliverable: probe
module + score comparison notebook.

**Week 3 — Tournament v1 (emulation).**
Implement mask assembly (blend + diversity caps) and the short-run DE
tournament across all mask families × 3 seeds × ≥2 eval sets. Deliverable:
paired results table, loss-vs-evals curves, first verdicts on hypotheses 1–5.

**Week 4 — Interpretation + hardware validation.**
Feature regression on tournament logs; extract candidate heuristics. Port the
top 2–3 mask families to hardware and re-run short tournaments on the PL to
confirm emulation-to-hardware transfer (the single most important validity
check). Deliverable: heuristic rule set + transfer report.

**Week 5 — Full tuning runs.**
Full-budget (≥300 evals) DE runs on hardware with the best heuristic masks vs
random-mask control, multiple seeds, held-out local validation. Optionally:
warm-started DE using probe `direction`. Deliverable: headline accuracy /
convergence results.

**Week 6 — Write-up + robustness.**
Ablations (mask size sweep, bitwidth if available, layer choice), drift
re-scoring test (perturb the "local" distribution, check score reuse), report.

Risk buffer: if the emulator cannot be made bit-accurate in week 1, fall back
to hardware-only probing with the reduced budget in §4 and shrink the
tournament to 3 mask families; the pipeline is unchanged, only T and the
number of comparisons scale down.

---

## Appendix — Interfaces to Existing Code

- Mask enters the existing loop via `self.candidates` (boolean array, layer
  shape) — the identification program's output is exactly this array (plus an
  optional `direction` array for warm starts), so no changes to
  `discrete_de_tuning_loop` are needed beyond accepting a supplied mask and
  optional initial population.
- Probe evals on hardware reuse the identical write path:
  reconstruct full array → `pack_layer(...)` → `load_runtime_weights()` →
  `pl_acc_eval()`; keep `starting_weights_dir` read-only as now.
- Nevergrad optimiser alternatives previously shortlisted for comparison:
  `DiscreteDE`, `DiscreteOnePlusOne`, `DiscreteLenglerOnePlusOne`,
  `FastGADiscreteOnePlusOne`, `LhsHSDE` — mask tournament and optimiser
  comparison are orthogonal axes; fix the optimiser (DiscreteDE) while
  comparing masks, then revisit.
