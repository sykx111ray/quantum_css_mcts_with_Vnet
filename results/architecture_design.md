# Value Network Architecture Redesign

## Matrix-Aware Encoding for Quantum CSS Code Synthesis

**Date:** 2026-06-29
**Project:** V-Net for MCTS-based Quantum Circuit Synthesis
**Label:** Minimum of 50 rollouts (fixed)
**Current architecture:** MLP(flattened_matrix) — flat vector input, no structural awareness

---

## 1. Problem Analysis

### 1.1 Current Bottleneck

The existing MLP achieves R² = 0.97 on Steane (7 qubits), 0.94 on RM15 (15 qubits), but only 0.82 on Surface d=5 (25 qubits). MCTS performance degrades catastrophically: V-Net MCTS is 172% worse than rollout MCTS on Surface.

Root cause: flattening a binary stabilizer matrix into a 1D vector destroys all structural information:
- Row identity (which stabilizer)
- Column identity (which qubit)
- Row-column interactions (which qubits each stabilizer couples)
- Sparsity patterns (each stabilizer involves only 2-4 qubits)

### 1.2 Structural Properties of the Input

The input is a binary matrix M x N where:
- **Rows** = X-type stabilizer generators (constraints on the quantum state)
- **Columns** = data qubits (physical quantum bits)
- **M[i,j] = 1** means stabilizer i involves qubit j
- The matrix is sparse: for Surface codes, each row has weight 2-4
- Row operations (Gaussian elimination) transform rows linearly over GF(2)
- Column operations (CNOT gates) correspond to GF(2) column additions
- Row permutation is physically meaningless (stabilizer ordering is arbitrary)
- Qubit permutation has physical meaning (connectivity) but is partially symmetric

### 1.3 What the Value Function Must Capture

For sibling ranking, the V-Net must detect subtle differences between similar matrices. Two sibling states differ by one CNOT operation — typically affecting only 1-2 columns. The architecture must:
1. Detect local column patterns (which qubits are "heavy")
2. Model row-column interaction (which stabilizer-qubit pairs are active)
3. Be sensitive to column permutations (qubit identity matters for routing cost)
4. Be **invariant** to row permutations (stabilizer ordering is arbitrary)
5. Generalize across different matrix sizes (Steane 3x7 → Surface 12x25 → Surface 24x49)

### 1.4 Performance Constraints for MCTS

Current MLP inference: 0.02 ms/call. MCTS performs thousands of evaluations per search.
- Target: <0.1 ms/call (still 10x faster than 50-rollout evaluation at 0.98 ms)
- Hard constraint: <0.5 ms/call (must remain faster than single rollout)
- Batch inference not applicable (MCTS evaluates one node at a time)

---

## 2. Candidate Architectures

### 2.1 Architecture A: Conv2D Matrix Encoder

**Design:**
```
Input: M x N binary matrix [1 x M x N]
  -> Conv2d(1, 32, kernel=3, padding=1) + ReLU
  -> Conv2d(32, 64, kernel=3, padding=1) + ReLU
  -> Conv2d(64, 128, kernel=3, padding=1) + ReLU
  -> AdaptiveAvgPool2d(1) -> [128]
  -> Linear(128, 64) + ReLU
  -> Linear(64, 1)
```

**Key properties:**
- Naturally handles variable input size via AdaptiveAvgPool
- Captures local 2D patterns (e.g., clusters of 1s)
- Translation equivariance across both rows and columns — columns are spatially meaningful but rows should be permutation invariant (CON: CNN is not row-permutation invariant)
- Row permutation creates different convolution outputs → **violates physical symmetry**

**Parameters (Surface d=5):** ~30K
**Inference estimate:** ~0.03 ms

### 2.2 Architecture B: Row-wise Transformer + Mean Pool

**Design:**
```
Input: M x N binary matrix
  -> Row Embedding: Linear(N, d_model=64) for each row [M x 64]
  -> + Learnable Row Position Encoding [M x 64] (optional, for ordering awareness)
  -> TransformerEncoder(2 layers, 4 heads, d_model=64, d_ff=128)
     Self-attention across M rows
  -> Mean pool over M rows -> [64]
  -> Linear(64, 32) + ReLU
  -> Linear(32, 1)
```

**Without row position encoding:** Fully row-permutation invariant (attention is permutation-equivariant, mean is permutation-invariant). This is physically correct — stabilizer ordering is arbitrary.

**With row position encoding:** Breaks permutation invariance but allows the network to learn if ordering matters for the heuristic solver (the solver has a fixed row processing order).

**Key properties:**
- Row permutation invariant (desirable, without positional encoding)
- Captures cross-row correlations (which stabilizers share qubits)
- Handles variable M naturally (attention over variable-length sequence)
- O(M²) attention complexity — negligible for M ≤ 50

**Parameters (Surface d=5):** ~40K
**Inference estimate:** ~0.05 ms

### 2.3 Architecture C: Dual-Stream Row-Column Transformer

**Design:**
```
Input: M x N binary matrix

Row Stream:
  -> Row tokens: Linear(N, d_model=64) -> [M x 64]
  -> Self-attention across M rows (2 layers)
  -> Mean pool -> [64]

Column Stream:
  -> Column tokens: Linear(M, d_model=64) -> [N x 64]
  -> Self-attention across N columns (2 layers)
  -> Mean pool -> [64]

Fusion:
  -> Concat([row_pool, col_pool]) -> [128]
  -> Linear(128, 64) + ReLU
  -> Linear(64, 1)
```

**Key properties:**
- Row permutation invariant in row stream
- Column attention captures qubit-qubit correlations
- Column permutation is NOT invariant — columns carry qubit identity (desirable: qubit label matters for routing cost)
- Handles variable (M, N) by design
- O(M² + N²) attention — for Surface d=7 (24² + 49²) = 2977 operations per attention layer, negligible

**Parameters (Surface d=5):** ~55K
**Inference estimate:** ~0.06 ms

### 2.4 Architecture D: Sparse Bilinear Encoder

**Design:**
This architecture is directly motivated by the GF(2) bilinear structure of the problem.
The value function depends on row-column interactions at active entries (M[i,j]=1).

```
Input: M x N binary matrix

Row Encoding (shared across all rows):
  -> Each row (N-dim binary vector) -> Linear(N, 32) + ReLU -> [M x 32]

Column Encoding (shared across all columns):
  -> Each column (M-dim binary vector) -> Linear(M, 32) + ReLU -> [N x 32]

Sparse Bilinear Interaction:
  -> For each active entry (i,j) where M[i,j]=1:
       interaction[i,j] = row_emb[i] * col_emb[j]  (elementwise)
  -> Sum over all active entries -> [32]
  -> (This is equivalent to: (M @ col_emb) * row_emb summed, exploit sparsity)

Global features:
  -> Row degrees: Linear(M, 16) applied to col sums -> [16]
  -> Column degrees: Linear(N, 16) applied to row sums -> [16]

Fusion:
  -> Concat([interaction_pool, row_deg, col_deg]) -> [64]
  -> Linear(64, 32) + ReLU
  -> Linear(32, 1)
```

**Key properties:**
- Explicitly sparse: only iterates over active entries (M[i,j]=1). For Surface d=5 (12x25 with ~3 ones per row = 36 active entries), this is extremely efficient.
- Row permutation invariant (summation over rows)
- Column permutation NOT invariant (each column has its own embedding)
- Naturally handles variable size via linear layers with variable input dim
- Captures the essential bilinear interaction: stabilizer-qubit coupling

**Challenge:** Linear(N, 32) and Linear(M, 32) have weights tied to specific dimensions. To handle variable N, we need either:
- (a) Separate row/col encoders per code size → not generalizable
- (b) Learned position-dependent embeddings with max size padding → wastes parameters
- (c) 1D convolution over columns/rows → generalizes to any size

**Variant D' (Size-general):** Use 1D Conv for row/col encoding instead of Linear:
```
Row encoding: Conv1d(in_channels=1, out_channels=32, kernel_size=3) over N columns
  -> applied to each row independently -> [M x 32]
Column encoding: Conv1d(in_channels=1, out_channels=32, kernel_size=3) over M rows
  -> applied to each column independently -> [N x 32]
```
This generalizes to any matrix size because convolution is size-agnostic.

**Parameters (Surface d=5):** ~10K (very compact)
**Inference estimate:** ~0.02 ms (fastest candidate)

### 2.5 Architecture E: Set Transformer (ISAB) with Column-CNN Rows

**Design:**
```
Input: M x N binary matrix

Row Encoding (size-agnostic):
  -> Conv1d(1, 32, kernel_size=3) applied to each row independently -> [M x 32]
  -> Conv1d(32, 64, kernel_size=3) -> [M x 64]

Set Transformer (Induced Set Attention Block):
  -> ISAB with I=4 inducing points
  -> 2 ISAB layers
  -> Self-attention within the set of rows, but reduced to O(I*M) from O(M²)
  -> Output: [M x 64]

Pooling:
  -> Mean pool over M rows -> [64]
  -> Max pool over M rows -> [64]
  -> Concat -> [128]

Head:
  -> Linear(128, 64) + ReLU + Dropout(0.1)
  -> Linear(64, 1)
```

**Key properties:**
- Row permutation invariant (Set Transformer is permutation equivariant, mean/max pool is invariant)
- ISAB reduces attention complexity from O(M²) to O(I·M), relevant for very large codes (M > 100)
- Column structure captured by 1D convolutions
- Inducing points learn prototypes of row patterns
- Overkill for current code sizes (M ≤ 50) but future-proof

**Parameters (Surface d=5):** ~50K
**Inference estimate:** ~0.08 ms

---

## 3. Architecture Comparison

### 3.1 Quantitative Comparison

| Criterion | A: CNN-2D | B: Row Transf. | C: Dual Transf. | D: Sparse Bilinear | E: Set Transf. |
|-----------|-----------|----------------|-----------------|--------------------|----------------|
| **Parameters** | ~30K | ~40K | ~55K | ~10K | ~50K |
| **Inference (ms)** | ~0.03 | ~0.05 | ~0.06 | ~0.02 | ~0.08 |
| **Row perm. invariance** | No (WRONG) | Yes (w/o PE) | Yes (row stream) | Yes | Yes |
| **Column sensitivity** | Yes | Limited | Yes | Yes | Moderate |
| **Variable M x N** | Yes | Yes (M var) | Yes | Yes (D' variant) | Yes (M var) |
| **Sparsity exploitation** | Partial | No | No | Yes (explicit) | No |
| **Cross-row attention** | Local only | Global | Global (row) | No | Global |
| **Cross-column attention** | Local only | No | Global (col) | No | No |
| **Implementation complexity** | Low | Medium | Medium | Low | Medium-High |
| **Scalability (M > 50)** | Good | O(M²) issue | O(M²+N²) issue | Excellent | O(I·M) good |

### 3.2 Qualitative Assessment

**A (CNN-2D):** **Not recommended.** The row-permutation sensitivity is a fundamental flaw. The same quantum code represented with different stabilizer orderings produces different CNN outputs. A V-Net trained on one ordering may fail when MCTS enumerates children (which have arbitrary row orderings). This is a hard failure mode.

**B (Row Transformer):** Viable but limited. Captures row-row interactions well but treats column structure only through the initial row embedding (which is a linear projection of the entire row). Column-level patterns (which qubits are "heavy") are implicit in the embedding but not explicitly modeled.

**C (Dual Transformer):** Most expressive. Models both row-row and column-column interactions explicitly. The column attention is particularly valuable because CNOT operations affect columns — a sibling state differs by column operations. Column attention can learn to compare qubit patterns.

**D (Sparse Bilinear):** Most theoretically elegant. Directly models the GF(2) bilinear structure: value = f(row_embeddings, column_embeddings, active_entries). The sparsity exploitation means it's the fastest candidate and scales best to large codes. However, it lacks explicit cross-row or cross-column attention — interactions are mediated only through the bilinear product.

**E (Set Transformer):** Most sophisticated set-based architecture. The ISAB mechanism is designed for permutation-invariant set processing. However, for current code sizes (M ≤ 25), the regular Transformer (B or C) is more appropriate — ISAB's inducing points are beneficial only when M >> I.

### 3.3 Ranking by Expected Research Value

| Rank | Architecture | Rationale |
|------|-------------|-----------|
| **1** | **C: Dual Transformer** | Best balance of expressiveness, structural correctness, and practicality. Column attention directly models the CNOT operation space. Row permutation invariance is correct. Demonstrated effectiveness of Transformers on structured inputs makes this a strong research contribution. |
| **2** | **D: Sparse Bilinear** | Most elegant theoretical motivation. If it works, it's highly publishable (GF(2)-aware architecture, sparsity exploitation). Risk: may be too simple — lacking cross-row attention might limit expressiveness. |
| **3** | **B: Row Transformer** | Simpler version of C. Lower risk, lower reward. Good baseline for Transformer-based approaches. |
| **4** | **E: Set Transformer** | Overengineered for current problem scale. Higher implementation complexity without proportionate benefit. Research value emerges only at very large codes (M > 100). |
| **5** | **A: CNN-2D** | Flawed by design (row-permutation sensitivity). Not recommended as primary architecture. May serve as a baseline to demonstrate that permutation matters. |

---

## 4. Recommended Architecture: Dual-Stream Row-Column Transformer (C)

### 4.1 Justification

1. **Structural correctness:** Row stream is permutation-invariant (stabilizer ordering is arbitrary). Column stream preserves column identity (qubit labels matter). This correctly models the physical symmetries.

2. **CNOT awareness:** In quantum circuit synthesis, CNOT gates operate on pairs of columns. Column self-attention can learn to compare column patterns — directly relevant to the sibling ranking task where siblings differ by column operations.

3. **Variable-size handling:** Both streams naturally handle variable M and N. The row encoding `Linear(N, d_model)` can be replaced with a 1D Conv for generalization across N if needed, but for fixed code families, learning separate encoders per size is acceptable.

4. **Proven paradigm:** Transformers have demonstrated strong performance on structured inputs (molecules, graphs, code). The dual-stream design is analogous to successful architectures in computational biology (protein structure prediction) and graph learning.

5. **Practical inference speed:** Estimated 0.06 ms/call — 16x faster than 50-rollout evaluation, 3x slower than current MLP. Well within the <0.1 ms target.

6. **Research contribution:** The specific application of dual-stream Transformers to GF(2) stabilizer matrices is novel. The row/column decomposition is mathematically motivated (GF(2) bilinearity) and practically effective.

### 4.2 Architecture Details

```
V-Net v2: Dual-Stream Transformer Value Network
================================================

Input: Binary matrix M x N (stabilizers x qubits)

Hyperparameters:
  d_model = 64
  n_heads = 4
  n_layers = 2
  d_ff = 128
  dropout = 0.0 (no dropout for deterministic MCTS inference)

Row Stream:
  Input [M, N] -> Linear(N, d_model) -> [M, d_model]
  -> Learnable RowType Embedding [M, d_model] (encodes "row index" as stabilizer type)
     (Optional: remove for strict row-permutation invariance)
  -> TransformerEncoderLayer x 2:
       MultiheadAttention(M tokens, 4 heads)
       FeedForward(d_ff=128)
       LayerNorm + Residual
  -> Mean pool over M tokens -> [d_model]

Column Stream:
  Input [N, M] (transpose) -> Linear(M, d_model) -> [N, d_model]
  -> Learnable Column Position Encoding [N, d_model]
  -> TransformerEncoderLayer x 2:
       MultiheadAttention(N tokens, 4 heads)
       FeedForward(d_ff=128)
       LayerNorm + Residual
  -> Mean pool over N tokens -> [d_model]

Fusion:
  -> Concat[row_pool, col_pool] -> [2 * d_model] = [128]
  -> Linear(128, 64) + ReLU
  -> Linear(64, 32) + ReLU
  -> Linear(32, 1) -> scalar value

Total parameters (Surface d=5: M=12, N=25, d_model=64):
  Row embedding: 25 * 64 = 1,600
  Column embedding: 12 * 64 = 768
  Row Transformer (2 layers): ~66K
  Column Transformer (2 layers): ~66K
  Fusion MLP: ~10K
  Total: ~145K parameters
  (Compared to baseline MLP: 3.5K on Steane, 50K on Surface)

Output normalization: z-score normalization as in baseline.
```

### 4.3 Computational Complexity

For a matrix M x N:

| Operation | Complexity | Surface d=5 (12x25) | Surface d=7 (24x49) |
|-----------|-----------|---------------------|---------------------|
| Row attention | O(M² · d) | 144 · 64 = 9.2K ops | 576 · 64 = 37K ops |
| Column attention | O(N² · d) | 625 · 64 = 40K ops | 2401 · 64 = 154K ops |
| Row FFN | O(M · d · d_ff) | 12 · 64 · 128 = 98K | 24 · 64 · 128 = 197K |
| Column FFN | O(N · d · d_ff) | 25 · 64 · 128 = 205K | 49 · 64 · 128 = 402K |
| Total ~ops per forward | | ~350K | ~790K |
| Estimated time | | ~0.06 ms | ~0.12 ms |

Even at Surface d=7 (49 qubits), inference is ~0.12 ms — still 8x faster than 50-rollout evaluation.

### 4.4 Design Decision: RowType Embedding

The Row Transformer has an optional learnable row-type embedding. This is a deliberate design choice:

**With row-type embedding:** The network can learn that certain stabilizer positions (e.g., boundary vs. bulk stabilizers in Surface codes) have different characteristics. This may improve performance on structured codes where row position carries physical meaning.

**Without row-type embedding:** Strict row-permutation invariance. The network processes rows purely based on their binary content, not their index. This is theoretically cleaner but may lose some position-dependent information.

**Recommendation:** Start without row-type embedding (strict invariance). Add it only if the ablation shows it helps. Column position encoding should always be used because qubit positions carry physical meaning (connectivity, boundary vs. bulk).

---

## 5. Training and Evaluation Plan

### 5.1 Training Protocol

Same as baseline to enable direct comparison:
- Dataset: 2000 train, 500 val, 500 test
- Label: Minimum of 50 rollouts
- Loss: MSE (keep same as baseline for direct comparison)
- Optimizer: Adam, lr=1e-3, ReduceLROnPlateau
- Early stopping: patience=30
- Input normalization: no normalization (binary matrix needs no scaling)
- Target normalization: z-score (same as baseline)

### 5.2 Evaluation Metrics

Beyond regression metrics (R², MAE, RMSE), evaluate:
1. **Ranking consistency** (Steane + Surface): Spearman ρ, Top-1/Top-3 accuracy
2. **Cross-code generalization:** Train on Steane, test ranking on Surface (zero-shot)
3. **Closed-loop MCTS:** Full MCTS integration on both codes
4. **Inference speed:** Microbenchmark vs baseline MLP
5. **Ablation studies:**
   - Row-only Transformer (B) vs Dual Transformer (C)
   - With vs without row-type embedding
   - With vs without column position encoding

### 5.3 Minimum Viable Experiment

The first experiment should be:
1. Implement Dual Transformer for Surface d=5 (12x25 matrix)
2. Train on Minimum label
3. Evaluate ranking consistency
4. Compare against baseline MLP on same data

If ranking consistency improves significantly (Spearman from 0.44 → 0.6+), the architecture is validated and can be refined.

---

## 6. Implementation Notes

### 6.1 Key Implementation Decisions

1. **Shared row embedding weight across rows:** The Linear(N, d_model) embedding is shared — the same projection is applied to every row. This is critical for permutation equivariance and size generalization.

2. **No batch norm, use layer norm:** Layer norm is standard in Transformers and doesn't need batch statistics, which is important for MCTS inference (batch size = 1).

3. **No dropout during inference:** Disable dropout for MCTS evaluations. Use dropout=0 during inference.

4. **Padding-aware attention:** For codes with variable M (different numbers of stabilizers across codes with the same architecture), use a padding mask in attention to ignore padded rows.

5. **Parameter sharing across codes:** For the initial implementation, create separate models per code size (different Linear(N, d_model) dimensions). For future generalization, use Conv1D embedding that works for any N.

### 6.2 Code Structure

```
architecture_v2.py:
  class DualTransformerVN(nn.Module):
    - RowTransformer(M, N, d_model, n_heads, n_layers)
    - ColumnTransformer(M, N, d_model, n_heads, n_layers)
    - FusionHead(2*d_model, hidden, 1)
    
  class RowTransformer(nn.Module):
    - self.embed = nn.Linear(N, d_model)  # or Conv1d for variable N
    - self.row_type_emb = nn.Embedding(max_M, d_model)  # optional
    - self.encoder = TransformerEncoder(...)
    
  class ColumnTransformer(nn.Module):
    - self.embed = nn.Linear(M, d_model)  # or Conv1d for variable M
    - self.pos_emb = nn.Embedding(max_N, d_model)
    - self.encoder = TransformerEncoder(...)
```

### 6.3 Comparison Skeleton

For the comparison experiment:
```python
architectures = {
    "mlp_flat": MLP(flattened_input, [64, 32]),
    "cnn_2d": Conv2DEncoder(M, N),
    "row_transformer": RowTransformerVN(M, N, d_model=64),
    "dual_transformer": DualTransformerVN(M, N, d_model=64),
    "sparse_bilinear": SparseBilinearVN(M, N, d_embed=32),
}
```

---

## 7. Why This Should Improve Sibling Ranking

### 7.1 The Core Mechanism

Sibling states differ by one CNOT operation. A CNOT adds column A to column B (mod 2) in the GF(2) matrix. This changes the column pattern of column B.

**Flat MLP:** The entire matrix is flattened. A change in one column affects entries at positions (i, B) for all rows i. These positions are scattered throughout the flat vector. The MLP must learn that positions (0,B), (1,B), ..., (M-1,B) are correlated. This is a distributed representation that requires many training examples.

**Dual Transformer:** 
- The column stream directly processes each column as a token. Column B sees its own full pattern (all M entries). Self-attention compares column B to other columns. A CNOT that changes column B is directly visible as a change in the column B token.
- The row stream sees which stabilizers are affected. Cross-row attention identifies that stabilizers sharing column B are now different.
- The fusion layer combines row-level and column-level information.

This explicit column-level processing should make sibling distinctions more learnable with fewer examples.

### 7.2 Generalization Mechanism

**Flat MLP on Surface d=5 → d=7:** The input dimension changes from 300 to 1176. A new model must be trained from scratch. No transfer is possible.

**Dual Transformer on Surface d=5 → d=7:** If the row/column embeddings use Conv1D (size-agnostic), the same model can process both sizes. The attention mechanism is independent of M and N. The architecture learns generic "stabilizer patterns" and "qubit patterns" that transfer across code sizes.

Even without Conv1D embeddings, the Transformer architecture can be initialized from a smaller model (embedding weights can be interpolated or discarded while attention weights transfer). This enables curriculum learning: train on small codes first, fine-tune on larger codes.

---

## Appendix A: Architecture Not Recommended (CNN-2D)

CNN-2D is explicitly not recommended because:

```python
# The same quantum state, with rows permuted:
matrix_original = np.array([[1, 0, 1, 0],   # stabilizer 1
                             [0, 1, 0, 1]])  # stabilizer 2

matrix_permuted = np.array([[0, 1, 0, 1],   # stabilizer 2
                             [1, 0, 1, 0]])  # stabilizer 1

# CNN-2D output: 
# conv_output(original) != conv_output(permuted)  # PROBLEM

# Row Transformer output (without row position encoding):
# transformer_output(original) == transformer_output(permuted)  # CORRECT
```

This is not just a theoretical concern. In MCTS, children are generated by applying actions to the parent state. The action set depends on the environment's row ordering. Different MCTS runs may encounter the same quantum state with different row orderings. A network that is not row-permutation invariant will produce inconsistent evaluations.

## Appendix B: Architecture Not Recommended (Set Transformer for Current Scale)

ISAB (Induced Set Attention Block) reduces attention complexity from O(M²) to O(I·M) by using I inducing points as a bottleneck. For M=12 (Surface d=5), M²=144 while I·M=48 (with I=4). But the overhead of the inducing point mechanism (additional cross-attention layers) outweighs the benefit at this scale. ISAB becomes beneficial only when M > 100, which corresponds to codes with 100+ stabilizers — far beyond current targets.
