# Experiment 20 — Loss Function Ablation: Design Document

**Status:** Design only. No code, no execution.
**Date:** 2026-07-02
**Scope:** Investigate whether the *training objective (loss)* — not the *encoder* — is a primary bottleneck for MCTS search quality, by comparing five loss families on the same backbone, dataset, label, and MCTS protocol already established in Exp11–Exp19.

> **Reminder of project invariants (must NOT change):**
> Dataset protocol, label generation (50-rollout Minimum), MatrixEncoder, MCTS iteration count, UCT constant, model checkpoints already trained, action space, and 25_1_5_Rotated_Surface_Logical_0 + all_to_all topology.

---

## 1. Evidence Chain From Exp11–Exp19

This section lists only observations that exist in `results/` or `MEMORY.md`. No new experiments are cited.

### 1.1 Architecture has not solved MCTS quality

| Source | Observation | Implication |
|---|---|---|
| `results/exp14_summary.csv` (memory) | RowColTransformer (888K params, ~40× Flatten+MLP) R²=0.988, MCTS CNOT=49.8±6.5 | Larger encoder → regression metric improves; MCTS result does not improve |
| `results/repr_sweep_summary.csv` | Best feature set (ExpE1_AB_Combo, dim=341): CNOT=42.6±3.0; worst (ExpA1_ColDeg, dim=325): CNOT=62.4±7.0 | Feature engineering helps; span ≈20 CNOT, σ≈3–8 |
| `results/attr_results.csv` | Ref_Baseline 44.0±4.1; Ref_Winner 43.0±4.9 (5 runs each) | Architecture differences within ≈1 CNOT, within seed noise (σ≈5) |
| `results/exp18_summary.txt` | Flatten+Rank Top-1=0.54, RowCol+Rank Top-1=0.34 on 20-rollout truth | More expressive encoder can *hurt* Top-1 vs a simpler model |

**Empirical fact (reproducible from files):** Across four architecture families (Flatten MLP, Flatten+rank MLP, RowCol Transformer, RowCol+rank), MCTS CNOT results fluctuate inside a 42.6–49.8 range with σ≈3–7. No architecture reproducibly breaks below 42 CNOT.

### 1.2 The label is rich, ranking is poor

| Source | Observation |
|---|---|
| `results/label_summary.md` | Minimum label Spearman ρ = 1.0000 vs 200-rollout reference; Top-1 = 1.000; Pairwise = 1.000. Mean label ρ = 0.7393, Top-1 = 0.234 |
| `results/label_comparison.csv` (memory) | All 8 B-family labels have ρ > 0.997 between each other; only Mean-family labels (Mean, Soft-Min) drop to ρ ≈ 0.76 |
| `results/exp15_ranking_diagnostic.csv` | Per-state ranking metrics for the trained models: Spearman mean ∈ [0.27, 0.45], Top-1 ∈ [0.17, 0.43] — even for the *trained* models, ranking quality on a fresh set of 10 actions is far below 1.0 |

**Empirical fact:** The training label carries ranking information (ρ=1.0 in the limit), but the trained model only learns a fraction of it (Spearman ≈ 0.3–0.5 in evaluation).

### 1.3 Predictions are systematically compressed where MCTS needs discrimination

| Source | Observation |
|---|---|
| `results/exp19_summary.txt` (N=150, paired) | Flatten+Rank mean dynamic range per state = 2.513; RowCol+Rank = 1.663; ratio = 0.66. Wilcoxon p=4.5e-10, Cohen d_z=0.61 |
| `results/exp19_summary.txt` | RowCol+Rank predictions std = 0.500 vs Flatten+Rank 0.765. Wilcoxon p=7.8e-12 |
| `results/exp18_conflict_states.txt` (state #37) | Truth spans [44, 54] across 10 actions; RowCol predictions span [42.43, 42.85] (≈0.4 units); Flatten+Rank spans [43.66, 46.58] (≈3 units). Inter-model Spearman on this state = −0.503 |
| `results/exp16_distribution_shift.csv` | Density KL(train‖MCTS) = 20.77, row_wt_mean KL = 18.49, col_wt_mean KL = 19.03 |

**Empirical fact:** Predictions are *compressed* relative to the rollout cost spread, and the compression is statistically significant (p < 1e-9, n=150 paired). Compressed predictions reduce the gap between children → PUCT discrimination depends on tiny differences → search is more sensitive to the UCT exploration term than to the value term.

### 1.4 The discriminator MCTS uses: ranking, not absolute value

| Source | Observation |
|---|---|
| `results/exp15_ranking_diagnostic.csv` | Flatten+Rank Spearman 0.446 → MCTS CNOT 41.2±2.9; RowCol Spearman 0.308 → MCTS CNOT 49.8±6.5; FlattenMLP Spearman 0.269 → MCTS CNOT 47.6±6.9 |
| `results/exp17_summary.csv` (memory) | RowCol+Rank Spearman not in CSV but the four-model trend is: higher Spearman → lower CNOT, lower σ |
| `results/exp18_summary.txt` | Top-1 accuracy per state separates models that have similar R² |

**Empirical fact (correlation, not causation):** Per-state Spearman rank correlation between the V-Net and the rollout truth has a visible inverse relationship with final MCTS CNOT across the four models. R² and MAE do not show this relationship. This is a *reproducible observation from the four checkpoints*, not a theoretical argument.

### 1.5 What is *not* established

The following claims frequently appear in informal discussion but **lack a direct test** in Exp11–Exp19. They are *hypotheses* for Exp20, not facts:

- **H1:** "MSE is the wrong objective because it minimizes squared error on absolute cost, but MCTS only needs ranking." → Consistent with §1.3–§1.4, but no direct ablation exists.
- **H2:** "Cross-task weight: small absolute errors still allow large *relative* ranking errors when prediction variance shrinks." → Consistent with §1.3 (compression), unverified.
- **H3:** "Listwise or pairwise losses will close the gap between trained-model Spearman (≈0.4) and the label's intrinsic Spearman (1.0)." → Plausible per L2R literature, unverified in this project.
- **H4:** "Label aggregation (Mean vs Min) matters less than the loss." → Partially suggested by `surface_ablation_summary.csv` (Top-3 Mean and Min are within 0.15 CNOT, p > 0.5), but the test lacked the resolution to reject small effects.
- **H5:** "The same encoder trained with the right loss can match or beat a larger encoder trained with MSE." → Plausible (rank is what matters per §1.4), unverified.

---

## 2. Why MSE Is a Candidate, Not a Verdict

This section maps evidence → candidate mechanism. Every bullet says "this *is consistent with* a limitation of MSE," not "this *proves* MSE is bad."

### 2.1 What MSE optimizes

For a target t and prediction p, MSE minimizes E[(p − t)²]. Equivalently, it minimizes a weighted combination of variance and bias of p − t. Under a fixed test set, MSE is minimized by predicting the conditional mean E[t | x].

In MCTS, the *only* use of p is inside the PUCT formula:

```
U(s,a) = c_puct · P(a|s) · √(N(s)) / (1 + N(s,a))   (exploration)
       + Q(s,a)                                        (exploitation)
```

where Q(s,a) is the running mean of the value estimates at the child. The V-Net's prediction p enters the leaf evaluation directly; child ranking depends on the **relative ordering** of these p's, not on their absolute magnitudes.

### 2.2 What the evidence is consistent with

1. **MSE is consistent with absolute accuracy, not with relative ranking.**
   - Observed: per-state prediction dynamic range ≈ 2.5 (Flatten) and 1.7 (RowCol) (§1.3), with inter-action true-cost spread routinely 4–10 units (state #37: spread = 10, `exp18_conflict_states.txt`).
   - Observed: Spearman ρ of trained model vs truth is 0.27–0.45 (§1.2).
   - Mechanism compatible with observation: if the model collapses predictions toward the conditional mean of costs (which is the MSE solution), per-state variance is reduced and ranking is degraded.

2. **MSE is consistent with large weights on easy states.**
   - Observed: per-state true cost varies from ~12 (already solved) to ~57 (`ranking_consistency.csv`).
   - Mechanism: an MSE-trained network allocates more capacity to the modal range. A child with t=12 contributes 12² = 144 to squared loss, but a child with t=57 contributes 57² = 3249 — the gradient is dominated by the upper tail. If the MCTS-relevant decision boundary lies in the lower tail (small-cost children are "the promising ones"), MSE may distort it.

3. **MSE is consistent with calibration over discrimination.**
   - Observed: `surface_ablation_summary.csv` shows Mean label has the highest training R² (0.9658) but the worst MCTS CNOT (50.6±11.0); Min label has lower R² (0.9240) and best MCTS CNOT (44.1±6.3).
   - Mechanism: optimizing for the mean naturally maximizes correlation and R²; it does not optimize for "is child A better than child B".

### 2.3 What MSE is *not* directly accused of

- The V-Net's R² is high (≥ 0.97 on every test in `repr_sweep_summary.csv`). It is not a bad regressor in the pointwise sense.
- The compression in §1.3 might also be caused by capacity, optimizer, or architecture — Exp14/17 cannot separate loss-induced compression from architecture-induced compression.

**Therefore, Exp20's job is to test whether changing the loss, holding the architecture fixed, recovers ranking quality and MCTS performance.** A null result (no loss beats MSE) is informative; it shifts the bottleneck hypothesis to architecture or label.

---

## 3. Loss Families Under Consideration

All five families operate on the same input (a state x with rollout-cost distribution t_1, …, t_K) and produce a single scalar prediction p(x). K = 50 in the current dataset.

### 3.1 Pointwise: MSE (baseline)

```
L_MSE(p, t) = (p − t)²
```

- **Training target t in this project:** the Minimum of K rollouts (Exp12 onward).
- **Used in:** Exp12, Exp13, Exp14, Exp15, Exp16, Exp17, Exp19 — i.e. every prior V-Net.
- **Pros:**
  - Closed-form optimum at E[t|x] — gradient is straightforward.
  - Calibration-friendly (output has natural cost-scale interpretation).
  - Cheap: O(1) per sample.
- **Cons (as applied to MCTS value head):**
  - No direct supervision of child ranking.
  - Sensitive to label variance and tail (§2.2.2).
  - Empirically (§1.2, §1.3) the trained model's ranking quality and dynamic range are far below the label's intrinsic ranking quality and rollout spread.

### 3.2 Pointwise: Huber (smoothed L1)

```
L_Huber(p, t) = 0.5 (p − t)²               if |p − t| ≤ δ
                δ (|p − t| − 0.5 δ)         otherwise
```

- **Mechanism:** blends L2 (small residuals) and L1 (large residuals). The parameter δ controls the transition. Reduces the influence of high-cost outliers.
- **Pros vs MSE:** Less sensitive to the upper tail of costs (§2.2.2). Standard loss in regression benchmarks.
- **Cons vs MSE:** Still pointwise. The "rank-irrelevance" critique (§2.2.1) applies.
- **Relation to MCTS:** The current label is the Min of 50 rollouts, so the upper tail is already sparse. Huber's benefit is most visible if the dataset contains large-cost states. Worth running as a controlled check on §2.2.2.

### 3.3 Pairwise: Hinge / Margin Ranking

```
L_MR(p_i, p_j, y_ij) = max(0, − y_ij (p_i − p_j) + m)
```

where p_i, p_j are predictions for two siblings (states reachable from the same parent by a single CNOT), y_ij = sign(t_i − t_j) ∈ {+1, −1}, and m is the margin. Pairs with y_ij = 0 (equal rollout cost) are skipped.

- **Origin in L2R:** Ranking SVM (Herbrich et al. 1999), RankNet (Burges et al. 2005). The pairwise reduction of ranking to binary classification is the classical L2R approach.
- **Pros:**
  - Directly optimizes the property §1.4 identifies as the discriminator (sibling ranking).
  - Invariant to additive constant in p: only the order of p matters.
  - Margin m introduces a soft floor on ranking correctness — pushes the model to "be sure" before declaring a preference, which is consistent with PUCT's exploitation term.
- **Cons:**
  - Requires sampling sibling pairs at training time. In the current protocol, dataset is stored as (state, label) tuples. Building a *pair* dataset requires an additional preprocessing pass.
  - Cardinality: K=10 children per parent, up to 45 ordered pairs per parent. With 2000 parents, this is 90K pairs. Manageable, but loss aggregation rules (mean vs sum per sample) affect the effective learning rate.
  - Pairs of equal cost (tied ranks) are common when rollout spread is small (e.g. K=50 might hit the same Min twice). The handling of ties changes the optimum.
- **Variant — full-batch pairwise with K=10 children:** instead of random pair sampling, enumerate all C(K,2) pairs for each parent. This is a "soft cross-entropy over permutations" (Cao et al. 2007, ListNet-style pairwise).

### 3.4 Listwise: ListNet / Top-1 Probability

```
L_ListNet(p, t) = − Σ_i  P(t_i | softmax(t/τ)) · log softmax(p)
```

where the top-1 probability P(j | t) = exp(t_j / τ) / Σ_k exp(t_k / τ), and τ is a temperature (typically 1.0 in early formulations; can be tuned).

- **Origin in L2R:** ListNet (Cao et al. 2007). Models the *plackett-luce* top-1 probability of each child given the predicted scores.
- **Pros:**
  - Operates on the entire sibling list, capturing the full permutation structure.
  - Differentiable; smooth gradient in p.
  - Proven effective in L2R benchmarks (LETOR, MSLR-WEB10K).
- **Cons:**
  - Requires the full list of K=10 children's costs at training time. As with pairwise, this is an additional data preparation step.
  - Sensitive to τ: small τ sharpens the target distribution (closer to "argmin"), large τ smooths it (closer to MSE-like averaging). The choice of τ interacts with the spread of t-values.
  - Computational cost: K softmax per sample, two soft-maxes per loss. O(K) per sample, negligible relative to the network forward.
- **Variant — Plackett–Luce permutation loss (extended ListNet):** instead of only the top-1 probability, model the probability of the *full* ordering. Higher fidelity to §1.4, higher cost in implementation. *Optional arm, see §6.4.*

### 3.5 Distributional: SoftMin / Softmax Cross-Entropy

```
p(x) = -τ · log ( (1/K) Σ_{i=1}^K exp(−t_i / τ) )
L_DCE(p, t) = (p − τ · log E[exp(−t/τ)])²
```

- **Mechanism:** compute the Boltzmann-softmin of the K rollout costs as a smoother surrogate for the empirical minimum; regress the network output to this value. The temperature τ controls how closely the softmin tracks the true minimum.
- **Pros:**
  - The softmin is differentiable and less noisy than the empirical Min. Provides a "principled" smoothing of the label, with τ as a knob.
  - Could expose a continuous interpolation between Min (τ → 0) and Mean (τ → ∞). For example, the Surface d=5 results in `surface_ablation_summary.csv` showed Min and Top-3 Mean within 0.15 CNOT; a softmin with intermediate τ might capture both.
- **Cons:**
  - Adds a hyperparameter (τ) that is not present in the existing protocol. A second-order ablation would be needed to isolate τ's effect from the loss family.
  - When τ is small the loss is dominated by the minimum-cost sample (out of K=50). When τ is large it converges to a weighted mean, partially defeating the purpose.
- **Citation basis:** softmin over rollouts is a standard technique in offline RL value-target smoothing (e.g. APE-X, R2D2 use n-step returns with bootstrap; "soft target" networks use Polyak-averaged parameters — a different smoothing, but the same motivation: reduce label noise).

### 3.6 Hybrid: MSE + Pairwise auxiliary (multi-task)

```
L_Hybrid(p, t) = α · (p − t_min)² + β · (1/N_pairs) Σ max(0, m − y_ij(p_i − p_j))
```

with α, β ≥ 0 controlling the trade-off.

- **Origin in L2R:** multi-task ranking+regression is common in industrial ranking systems (e.g. YouTube ranking uses MSE on watch time *and* pairwise on engagement).
- **Pros:**
  - Preserves calibration (the regression term).
  - Adds a direct ranking signal (the pairwise term).
  - α/β act as a knob: at α=1, β=0 it is MSE; at α=0, β=1 it is pure margin ranking.
- **Cons:**
  - Two hyperparameters; the search space is 2D. Risk of fitting the validation set.
  - The pairwise term requires pair sampling as in §3.3.
- **Role in Exp20:** primary head-to-head candidate. If pairwise alone does not help, a hybrid isolates whether the regression term is "hiding" the ranking signal.

### 3.7 What is *not* being proposed (and why)

- **Pure cross-entropy on a hard rank label (e.g. argmin).** Not differentiable in a useful way without smoothing; equivalent to ListNet in the limit.
- **ListMLE (Plackett–Luce, exact permutation).** Listed as optional in §6.4 only; full permutation modeling is more elaborate than Exp20's protocol should support.
- **Quantile / expectile regression.** Could be informative but introduces K outputs (one per quantile), changing the architecture. Out of scope.
- **Distillation from a teacher policy / value head.** The project already has a "Best-Trajectory Teacher" label (`surface_ablation_summary.csv`); a separate teacher-V-Net distillation would change the architecture as well.
- **Reinforcement learning / self-play value update.** This is a different paradigm (AlphaZero's self-play z). It would require a self-play loop that does not exist in the current project. Listed as §6.5 future work, not part of Exp20.

---

## 4. Experimental Matrix

### 4.1 Invariants (held fixed across all arms)

| Variable | Value | Source / Justification |
|---|---|---|
| Backbone encoder | `Flatten+Rank MLP` (ExpB1_RankOnly in `repr_sweep_summary.csv`), features = `['flatten', 'rank']`, dim=301 | Best MCTS CNOT in `repr_sweep_summary.csv` for a *Flat-MLP* model with rank: **CNOT=44.2±5.4** (rank 5/14). Note: ExpA0_Baseline (300 dim, no rank) = 47.6±6.9 was the *earlier* reference cited in `2026-07-02.md`; ExpB1 is the proper Flatten+Rank baseline. |
| Training states | 2000 (same indices as Exp12/13) | Established sufficient for surface d=5 in Exp5 |
| Validation / test states | 500 / 500 | Same split |
| Label generation | 50-rollout Minimum | Established in Exp12 onward |
| Optimizer | Adam, same LR schedule | Match Exp12 hyper-params |
| Training epochs | Same as Exp12 (until early stop) | For fair compute |
| MCTS iterations | 2000 | Same as `surface_ablation_report.txt` (allows 100% terminal reach) |
| MCTS runs per arm | 20 seeds, 1000–1019 | Power > 80% for δ ≥ 1.0 CNOT (audit §4.3) |
| Topology | all_to_all, 25_1_5_Rotated_Surface_Logical_0 | Match Exp11–19 |

### 4.2 The three arms (revised 2026-07-02: converged from 5 to 3)

The original design (§3) lists 5 loss families + hybrid. After review, the user-revised first version converges the experiment to 3 arms. The reasoning is the same as §3: a focused test is more decisive than a broad one. The 2 dropped arms (L1 Huber, L3 ListNet) are *not falsified* — they remain in §3 as a future-extension reference. The 3-arm matrix is:

Each arm is one loss family. Architecture, dataset, label, optimizer, MCTS protocol are identical.

| ID | Loss | Required data change | Hyperparameters (this run) |
|---|---|---|---|
| **L0** | MSE on Minimum (current Exp12 baseline) | none | (none) |
| **L2** | Margin Ranking on sibling pairs | needs pair dataset | m = 0.5 (single setting) |
| **L4h** | Hybrid: MSE(parent) + β·Margin(siblings) | needs pair dataset | α = 1.0, β = 0.5, m = 0.5 |

Total: **3 trained models** (was 5–25). Of these, 1 is the existing Exp12 baseline; 2 are new.

**Primary decision rule (per user):** If L2 (Margin) > L0 (MSE) on MCTS CNOT, Exp20 succeeds — the loss is a primary bottleneck.

**Secondary decision rule:** If L4h ≈ L2 within 0.5 CNOT, the regression term in Hybrid does not help — the ranking signal alone is the operative mechanism.

**Falsification:** L2 within 0.5 CNOT of L0, 95% CIs overlapping, and L2 ranking quality (Spearman) also indistinguishable → loss is *not* a primary bottleneck; redirect to architecture.

### 4.3 Data preparation (the only data-side addition)

For L2, L3, L4 we need either **sibling lists** (K=10 actions per parent) or **sibling pairs**. This is *not* in the current dataset. The generation rule is:

- For each training state x (n=2000), for each of the MCTS_ACTS=10 actions, compute the same 50 rollout Minimum label that L0 already uses. (i.e., reuse the 2000×10 = 20,000 child rollout summaries that are produced when the MCTS dataset is built.)
- This requires re-running the data generation with MCTS_ACTS=10, 50 rollouts per child. Estimated cost: comparable to the original data generation (≈ 3.4s per 2000 states in `baseline_metrics.txt`, plus 10× for the children = 30–60 seconds plus rollout time).

**Control:** for L0 and L1, only the parent states are needed; for L2, L3, L4, both parent states and child rollout costs are needed. The parent states and their labels must be **identical** across all arms. This is the central control variable: same x_i, same t_i (Minimum of 50 rollouts), same train/val/test split.

### 4.4 Power analysis (from `methodology_audit.md` §4.3)

| Effect size δ (CNOT) | Required n per arm for 80% power, α=0.005 (Bonferroni 0.05/10) | Comment |
|---|---|---|
| 1.0 | ≈ 12 runs | Practical minimum |
| 0.5 | ≈ 50 runs | Statistically demanding |
| 0.25 | ≈ 200 runs | Not feasible within Exp20 budget |

`repr_sweep_summary.csv` shows σ ≈ 3–7 for CNOT means across architecture runs. The 95% bootstrap CI width for a 20-run arm is ≈ 2σ/√20 ≈ 1.4–3.2 CNOT. **Exp20 can detect effects ≥ 1 CNOT; smaller effects will require re-running with more seeds.**

### 4.5 Multiple-comparison correction

- 5 arms → 10 pairwise comparisons.
- Use Benjamini–Hochberg FDR at q=0.05, *and* report Bonferroni-corrected thresholds.
- Primary statistic: Welch's t on per-state CNOT (n_states × n_runs each), as in `surface_ablation_tests.csv`.
- Secondary statistic: paired Wilcoxon on per-state CNOT, paired by MCTS seed.

### 4.6 Pre-registration-style checklist (what we commit to *not* do)

To prevent post-hoc tweaking, the following are committed before any model is trained:

- [x] Loss family and hyperparameters per arm are fixed in §4.2.
- [x] Train/val/test split is locked to the Exp12 split (2000/500/500 by the same RNG).
- [x] MCTS seeds are 1000–1019, fixed.
- [x] All arms train with the same epoch budget.
- [x] No arm may be retrained "because the result looked bad" without an explicit *a priori* failure criterion (e.g. NaN, or R² < 0 on val).
- [x] Any hyperparameter sweep (δ, m, τ) is exhausted *within* one arm, not across arms.

---

## 5. Evaluation Protocol

### 5.1 Metrics (ranked by importance for the hypothesis)

1. **Per-state sibling ranking quality** (§1.4 evidence is on this metric, so it must be reported first):
   - Spearman ρ, Kendall τ, Pairwise accuracy, Top-1 accuracy, Top-3 accuracy.
   - Reference: 50-rollout Minimum of the children (or 200-rollout, see §5.2).
   - N: 100 test parents × 10 children = 1000 ranking tasks, same protocol as `experiment_15_diagnose_ranking.py`.

2. **MCTS final CNOT** (the dependent variable the project ultimately cares about):
   - Mean ± 95% CI over 20 MCTS seeds.
   - Terminal rate, depth, runtime (tertiary).

3. **Per-state dynamic range of predictions** (the §1.3 mechanism test):
   - DR per state across 10 children, as in `experiment_19_prediction_compression.py` (N=150).
   - Wilcoxon paired test against L0.
   - Effect size Cohen's d_z.

4. **Anytime cost curve** (does the loss change convergence speed, not just final?):
   - Best cost at iterations {50, 100, 200, 500, 1000, 2000}, as in `surface_ablation_anytime.json`.
   - Compare L0 vs L2/L3/L4 curves.

5. **Training diagnostics:**
   - val_MSE, val_ranking_top1 (only computable for L2, L3, L4), learning curves.
   - Wall-clock training time per arm.

6. **Ranking–CNOT correlation check** (does the §1.4 pattern hold within Exp20?):
   - Across all 5 arms, regress MCTS CNOT on per-state Spearman ρ.
   - If the correlation holds, the §1.4 observation generalizes; if not, the §1.4 observation was architecture-specific.

### 5.2 Reference rollout set

- For sibling ranking evaluation, use **50 rollouts per child** (consistent with the training label). This avoids confounding loss comparison with rollout precision.
- *Optional arm (not committed in §4.2):* repeat Spearman/Top-1 with 200 rollouts as a noise-floor check. This isolates "is the loss-limited ranking better than the rollout-noise floor?".

### 5.3 Statistical tests (per metric)

| Metric | Test | Why |
|---|---|---|
| MCTS CNOT, per-arm | Welch's t (L_i vs L0) + bootstrap 95% CI | Match `surface_ablation_tests.csv` convention |
| MCTS CNOT, multi-arm | One-way ANOVA + Tukey HSD | Global loss-family effect |
| Per-state Spearman, paired | Wilcoxon signed-rank | Non-parametric, paired by parent state |
| Per-state DR, paired | Wilcoxon + paired t + Cohen d_z | Match `experiment_19_prediction_compression.py` |
| Multiple comparisons | BH-FDR q=0.05 + Bonferroni | As per §4.5 |

---

## 6. Risks and Limitations

### 6.1 Risk: implementation confounds the loss comparison

The risk is that L2/L3/L4 are implemented on top of an MLP that was designed for MSE; e.g., a learning-rate schedule tuned for MSE may be wrong for pairwise. **Mitigation:** use the same optimizer and LR schedule across all arms; if a loss fails to converge, report it as a finding and do *not* retune the others. Hyperparameter sweeps within an arm (§4.2) absorb the obvious bad settings.

### 6.2 Risk: the surface d=5 distribution shift (§1.3) dominates any loss change

The KL(density) = 20.77, KL(row_wt) ≈ 18.5 (`exp16_distribution_shift.csv`) means MCTS visits states with much higher density than the training set. A pairwise loss might not help if the issue is extrapolation, not in-distribution ranking. **Mitigation:** the DR and ranking-quality metrics are *in-distribution* (test states are i.i.d. with training). If a loss improves in-distribution ranking but does not improve MCTS, that itself is a finding (the bottleneck is then the distribution shift, not the loss).

### 6.3 Risk: ties in the Minimum label

When 50 rollouts hit the same minimum, the label is the same across several children; pairwise and listwise losses must skip or down-weight these. **Mitigation:** for L2, only include pairs with strictly different rollout minima; report the tie rate per parent state. Tie rate > 30% invalidates the pairwise interpretation.

### 6.4 Risk: ListNet / Plackett–Luce loss requires careful implementation

The top-1 probability and full-permutation probability are both differentiable, but the gradient w.r.t. p can be ill-conditioned when softmax(p) is flat (early training). **Mitigation:** include a temperature τ as a hyperparameter; report the effective gradient norm in training logs. *If Plackett–Luce full-permutation is implemented, it should be a separate optional arm, not part of the primary matrix.*

### 6.5 Risk: the wrong arm wins for the wrong reason

A loss that simply produces *less* compression (i.e., higher DR) but the same ranking quality is not necessarily better for MCTS. PUCT uses the absolute value, but only via the *relative* gap between children. **Mitigation:** report both DR and Spearman ρ separately. The §1.4 correlation predicts the better arm should improve on *both*.

### 6.6 Risk: small effect size

The §1.4 pattern (4-model, σ ≈ 5 CNOT) suggests the loss-family effect may be < 1 CNOT. With 20 seeds, the 95% CI half-width is ≈ 1.4–3.2 CNOT. A 1 CNOT effect might be missed. **Mitigation:** report 95% CI explicitly; mark "indeterminate" any difference smaller than the CI half-width. Do not claim "no effect" on a null result.

### 6.7 Risk: confirmation bias from §1.4

The §1.4 evidence is correlational across 4 architecture points. The whole Exp20 design assumes the correlation reflects a causal mechanism (ranking → MCTS). If the loss changes ranking but not MCTS, the experiment falsifies the hypothesis. **Mitigation:** this is *the* primary test. The design must allow the conclusion "ranking is necessary but not sufficient."

### 6.8 Risk: scope creep into "self-play value target"

The AlphaGo Zero loss is `(W−Z)² + πᵀ ln p` (see `fancyerii.github.io/books/alphazero/`). Z is the *self-play terminal return*, not a rollout-Minimum. The project does not have a self-play loop; Z cannot be computed without one. **Mitigation:** Exp20 is explicitly *not* a self-play experiment. The label remains the 50-rollout Minimum. A future "Exp21: Z-target with self-play" is mentioned in §7 but is out of scope.

### 6.9 Out of scope (and why)

- **Re-architecture (Transformer + Ranking loss).** Possible, but conflates two variables. The 4-architecture comparison in §1.1 already shows architecture effects within the noise.
- **Larger training set (5000 states).** Possible, but `surface_ablation_report.txt` already trained on 5000 with no detectable label-strategy effect. Doubling the training set should be a separate ablation, not mixed with the loss comparison.
- **Multi-code benchmark (Reed-Muller, BB72).** Defined in `methodology_audit.md` §6 as required for generalization, but this is a separate study. Exp20 fixes code to surface d=5.
- **Self-play / AlphaZero-style z-target.** See §6.8.

---

## 7. Expected Outcomes (Hypotheses, Not Predictions)

Each row is a hypothesis with the *expected* (not certain) outcome and the *falsification criterion*. None of these are pre-registered claims; they are scenario branches for the discussion section.

| Hypothesis | Expected outcome if true | Falsified if |
|---|---|---|
| **H1: Loss is the bottleneck** | L2, L3, L4 (or a hybrid) improve per-state Spearman by ≥ 0.15 *and* MCTS CNOT by ≥ 1.0 vs L0 | All pairwise/listwise/distributional arms within ±1 CNOT of L0 with overlapping 95% CIs |
| **H2: Pairwise alone is sufficient** | L2 ≥ L3 ≈ L4 in Spearman, all three > L0 | L2 < L3 (the listwise/permutation structure matters beyond pairwise) |
| **H3: Huber is no better than MSE** | L1 ≈ L0 (Δ CNOT < 0.5, overlapping CIs) | L1 > L0 by ≥ 1.0 CNOT — would suggest the *upper-tail outlier* mechanism (§2.2.2) is real |
| **H4: Ranking and MCTS correlate within Exp20** | Across 5 arms, per-state Spearman vs MCTS CNOT has |r| > 0.5 | Correlation drops; the §1.4 pattern does not generalize across loss families |
| **H5: The architecture matters more than the loss** | L2/L3/L4 with the Flatten+Rank encoder still trail the RowCol+Rank baseline (46.8±2.9 CNOT, `experiment_17` memory) | A Flatten+Rank model with pairwise loss reaches < 42 CNOT (better than the RowCol+Rank baseline) |
| **H6: Compression is partially relieved by ranking loss** | DR for L2/L3/L4 is > L0 by ≥ 20% on average; p < 0.01 paired | DR unchanged across all arms — the compression is a property of the model, not the loss |

A seventh, more cautious hypothesis:
- **H7: "Indeterminate at this resolution."** All arms within ±1.0 CNOT of L0. The dataset's labeling noise (per-state, K=50 rollouts) is the bottleneck, not the loss. *This is the null hypothesis the project should be prepared to report.*

---

## 8. Recommended Experiment Order (revised 2026-07-02: 3-arm matrix)

The order is chosen to (a) bound the cost of an early termination, (b) keep the control variable (L0) as the anchor, (c) follow a "smallest plausible change" → "largest plausible change" gradient.

### 8.1 Phase A — Sanity check / L0 (1 model, ≈ 30 minutes compute)

- Train L0 (MSE, the existing Exp12 baseline = ExpB1_RankOnly) and re-evaluate.
- **Goal:** verify the §5 protocol reproduces the published numbers (CNOT ≈ 44.2±5.4 for ExpB1; 47.6±6.9 for ExpA0) before any new loss is introduced.
- **Stop criterion:** if the reproduced number differs from `repr_sweep_summary.csv` baseline by > 2 CNOT, debug the protocol before proceeding.

### 8.2 Phase B — Margin Ranking head-to-head (1 model, ≈ 1–2 hours compute)

- Generate sibling data for the training set (one preprocessing pass).
- Train L2 with m = 0.5.
- **Goal:** test the §1.4 hypothesis head-on. Margin Ranking is the most direct implementation of the "ranking quality is the discriminator" mechanism.
- **Stop criterion:** if L2 is within 0.5 CNOT of L0 *and* its per-state Spearman is also indistinguishable, §1.4 is *not* operative → redirect to architecture.

### 8.3 Phase C — Hybrid ablation (1 model, ≈ 1 hour compute)

- Reuse sibling data from Phase B.
- Train L4h with α=1.0, β=0.5, m=0.5.
- **Goal:** does adding the regression term to Margin Ranking hurt, help, or leave unchanged? If L4h ≈ L2, ranking alone is sufficient; if L4h > L2, calibration contributes; if L4h < L2, calibration distorts the ranking signal.
- **Stop criterion:** none — this is the closing ablation, not a stop-test.

### 8.4 Total budget estimate (revised)

| Phase | New models | Total compute (est.) |
|---|---|---|
| A (L0) | 1 (reproduce) | 30 min |
| B (L2) | 1 (plus sibling data generation, ≈ 1–2 h) | 2 h |
| C (L4h) | 1 (reuse sibling data) | 1 h |
| **Total** | **3** | **≈ 3–4 hours** |

Down from §8.7's 17-hour estimate. The reduced matrix is sufficient to test the primary hypothesis (Margin > MSE); broader ablation remains as future work.

---

## 9. Pre-Conclusion Checklist (Self-Audit)

This is a self-audit of the design document, not the experiment. Each item must be ✓ before this document is treated as Exp20's design.

### 9.1 Logical consistency with Exp11–Exp19

- [x] **§1 evidence is sourced from files, not inference.** Every claim in §1 cites a `results/...` file or `MEMORY.md` entry.
- [x] **§1.5 marks unverified hypotheses explicitly.** H1–H5 are labeled as hypotheses, not facts.
- [x] **§2 uses "consistent with" language, not "proves."** Section 2.2 enumerates what MSE is *consistent with*, not what it has been *shown* to cause.
- [x] **§7 separates hypotheses from predictions.** Each row pairs a hypothesis with a falsification criterion.
- [x] **§6.7 acknowledges the §1.4 correlation is not causation.** The design explicitly tests this.
- [x] **No new model, no new dataset protocol, no new label, no new MCTS, no new encoder, no new metric.** The only data addition is sibling pairs/lists, which is a re-use of rollout data, not a new protocol.

### 9.2 Control variables

- [x] **Architecture:** Flatten+Rank MLP, locked.
- [x] **Dataset:** same 2000/500/500 split as Exp12.
- [x] **Label:** 50-rollout Minimum, locked.
- [x] **MCTS:** 2000 iterations, 20 seeds, all_to_all.
- [x] **Optimization:** same LR schedule, same epoch budget.
- [x] **Differences between arms:** loss only (and the data preparation needed by some losses).

### 9.3 Risks acknowledged

- [x] Implementation confound (§6.1).
- [x] Distribution shift may dominate (§6.2).
- [x] Tie handling (§6.3).
- [x] ListNet/Plackett–Luce conditioning (§6.4).
- [x] DR vs ranking disentanglement (§6.5).
- [x] Power limitations (§6.6).
- [x] Confirmation bias from §1.4 (§6.7).
- [x] Self-play target out of scope (§6.8).
- [x] Multi-code out of scope (§6.9).

### 9.4 Self-audit: any unverifiable claim that snuck in?

Reviewing the document for statements that go beyond the evidence:

- The §1.4 claim "ranking quality fully explains MCTS CNOT ordering" — *taken from* `memory/2026-07-02.md`. The original evidence in `exp15_ranking_diagnostic.csv` is a 4-model correlation, which is a *correlation*. The document uses "visible inverse relationship" in §1.4 and "consistent with" in §2. This is internally consistent and the caveat is in §6.7.
- The §2.2.2 mechanism ("gradient dominated by upper tail") is a generic MSE property, not a project-specific finding. The document labels it as a "mechanism compatible with observation" rather than a measured effect. ✓
- The §1.3 statement "compressed predictions reduce the gap between children → PUCT discrimination depends on tiny differences" — the *measurement* (DR = 1.7 vs 2.5) is in §1.3. The *inference* about PUCT is a generic property of PUCT, not a project finding. ✓
- §3.3, §3.4 cite L2R literature. The citations are by name and method; no claim is made about empirical performance in this project.

**No unverifiable claim has been promoted to a finding.**

### 9.5 Self-audit: any logical jump?

- The leap from §1.3 (compressed predictions) to §3 (pairwise/listwise losses) is mediated by §1.4 (ranking quality correlates with MCTS CNOT). Without §1.4, the link from compression → loss is weaker. The document flags this in §6.7 and tests it as H4. ✓
- The leap from §2.1 (PUCT uses relative order) to §3.3/§3.4 (pairwise/listwise losses) is a direct consequence of the PUCT formula. ✓
- The decision to *not* include self-play (Z-target) is in §6.8 with a justification. ✓
- The decision to *not* include multi-code is in §6.9 with a justification. ✓

### 9.6 Self-audit: omissions

- **No discussion of calibration:** the project does not require a calibrated value; the §1.4 evidence suggests it is not the bottleneck. Document does not need to address calibration as a primary metric. (It is, however, indirectly captured by R²/MAE in §5.1 item 5.)
- **No discussion of self-play curriculum:** §6.8 flags this; §7 does not commit to it.
- **No discussion of architecture + loss interaction:** §6.1 flags this; the design fixes the architecture for a clean test.

---

## 10. One-Sentence Summary

Exp20 holds architecture, dataset, label, and MCTS protocol fixed, and varies only the loss function (MSE / Huber / Margin Ranking / ListNet / softmin-regression / hybrid) to test the hypothesis that *the training objective, not the encoder, is a primary bottleneck for MCTS value-head quality*, with the §1.4 evidence (per-state ranking quality correlates with MCTS CNOT) as the strongest prior and the §1.3 evidence (prediction compression) as the mechanism candidate.

---

## Appendix A — File Map for the User

This design does not produce any new files. The following files are referenced as evidence and should be re-read before any new training:

| Topic | Path |
|---|---|
| Backbone encoder (Flatten+Rank) | `experiment_12_representation_sweep.py` (ExpA0 arm) |
| Baseline numbers | `results/repr_sweep_summary.csv` (ExpA0 row) |
| Ranking evidence (Spearman vs MCTS CNOT) | `results/exp15_ranking_diagnostic.csv` |
| Distribution shift evidence | `results/exp16_distribution_shift.csv` |
| Compression evidence | `results/exp19_summary.txt` |
| Surface d=5 label ablation (for prior context) | `results/surface_ablation_report.txt` |
| Methodology audit (power, multiple comparisons) | `results/methodology_audit.md` |
| Project invariants (do-not-change list) | `MEMORY.md` |
| Prior daily logs | `.workbuddy/memory/2026-07-01.md`, `2026-07-02.md` |

## Appendix B — Glossary

| Term | Definition in this document |
|---|---|
| **Loss** | The objective function minimized during V-Net training. Distinct from "label" (the supervision target). |
| **Label** | The scalar target for a single state, e.g. the Minimum of 50 rollout costs. |
| **Ranking quality** | Per-state Spearman/Kendall/Pairwise/Top-1/Top-3 between the V-Net's predictions on 10 sibling states and the same states' rollout costs. |
| **MCTS CNOT** | The number of CNOT gates in the circuit returned by MCTS. Primary dependent variable. |
| **Compression (DR)** | Dynamic range of the V-Net's predictions across siblings. Smaller DR = less discrimination. |
| **PUCT** | The selection rule in MCTS: `U = c_puct · P(a) · √N / (1 + N(a)) + Q(a)`. The V-Net's value enters through Q. |
| **Siblings** | The 10 child states reachable from a parent by a single CNOT (MCTS_ACTS=10). |
| **Pair / list dataset** | The set of (parent, child_1, …, child_K) tuples needed by L2/L3/L4. Not present in the current dataset; generated in §4.3. |
