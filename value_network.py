"""
value_network.py — Value Network module for MCTS integration.
Provides the model class, normaliser, and matrix-to-input encoding.
Supports composable feature extraction via feature_config dict.
"""
import numpy as np
import torch
import torch.nn as nn

from matrix_encoder import MatrixEncoder


# ==============================================================================
# GF(2) linear algebra utilities (self-contained, no external deps)
# ==============================================================================
def _gf2_rank(matrix):
    """Compute rank of binary matrix over GF(2) via Gaussian elimination."""
    m = matrix.copy().astype(np.uint8)
    nrows, ncols = m.shape
    r = 0
    for col in range(ncols):
        pivot = None
        for row in range(r, nrows):
            if m[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != r:
            m[[r, pivot]] = m[[pivot, r]]
        for row in range(nrows):
            if row != r and m[row, col] == 1:
                m[row] ^= m[r]
        r += 1
        if r == nrows:
            break
    return r


def _gf2_rank_with_pivots(matrix):
    """Compute GF(2) rank AND return the number of pivot columns found.
    A pivot column is one that was selected as pivot during elimination."""
    m = matrix.copy().astype(np.uint8)
    nrows, ncols = m.shape
    r = 0
    for col in range(ncols):
        pivot = None
        for row in range(r, nrows):
            if m[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != r:
            m[[r, pivot]] = m[[pivot, r]]
        for row in range(nrows):
            if row != r and m[row, col] == 1:
                m[row] ^= m[r]
        r += 1
        if r == nrows:
            break
    return r, r  # rank, number of pivot columns found


def _row_similarity_pairs(matrix):
    """Compute pairwise row similarity (Jaccard-like: overlap/max weight).
    Returns (mean, max) across all row pairs."""
    nrows = matrix.shape[0]
    if nrows <= 1:
        return 0.0, 0.0
    row_weights = matrix.sum(axis=1)
    sims = []
    for i in range(nrows):
        wi = row_weights[i]
        if wi == 0:
            continue
        for j in range(i + 1, nrows):
            wj = row_weights[j]
            if wj == 0:
                continue
            overlap = int(np.dot(matrix[i], matrix[j]))
            sims.append(float(overlap) / max(wi, wj))
    if not sims:
        return 0.0, 0.0
    return float(np.mean(sims)), float(np.max(sims))


def _col_similarity_pairs(matrix):
    """Compute pairwise column similarity (overlap/max weight).
    Returns (mean, max) across all column pairs."""
    ncols = matrix.shape[1]
    if ncols <= 1:
        return 0.0, 0.0
    col_weights = matrix.sum(axis=0)
    sims = []
    for c1 in range(ncols):
        w1 = col_weights[c1]
        if w1 == 0:
            continue
        for c2 in range(c1 + 1, ncols):
            w2 = col_weights[c2]
            if w2 == 0:
                continue
            overlap = int(np.dot(matrix[:, c1], matrix[:, c2]))
            sims.append(float(overlap) / max(w1, w2))
    if not sims:
        return 0.0, 0.0
    return float(np.mean(sims)), float(np.max(sims))


# ==============================================================================
# Feature extractors — each returns a 1-D float32 numpy array
# ==============================================================================
# --- Group A: Basic statistics ---
def _feat_flatten(m):
    """Flatten(Matrix): raw binary matrix as 1-D vector."""
    return m.flatten().astype(np.float32)


def _feat_col_deg(m):
    """Column degrees: sum along rows → one value per column (qubit weight)."""
    return m.sum(axis=0).astype(np.float32)


def _feat_row_wt(m):
    """Row weights: sum along columns → one value per row (stabilizer weight)."""
    return m.sum(axis=1).astype(np.float32)


def _feat_density(m):
    """Matrix density: fraction of ones."""
    return np.array([float(m.sum()) / m.size], dtype=np.float32)


def _feat_sparsity(m):
    """Matrix sparsity: 1 - density."""
    d = float(m.sum()) / m.size
    return np.array([1.0 - d], dtype=np.float32)


def _feat_num_ones(m):
    """Total number of ones in the matrix."""
    return np.array([float(m.sum())], dtype=np.float32)


def _feat_col_deg_stats(m):
    """Column degree statistics: [min, mean, max]."""
    cd = m.sum(axis=0).astype(np.float32)
    return np.array([cd.min(), cd.mean(), cd.max()], dtype=np.float32)


def _feat_row_wt_stats(m):
    """Row weight statistics: [min, mean, max]."""
    rw = m.sum(axis=1).astype(np.float32)
    return np.array([rw.min(), rw.mean(), rw.max()], dtype=np.float32)


def _feat_col_deg_std(m):
    """Column degree standard deviation (spread of qubit participation)."""
    cd = m.sum(axis=0).astype(np.float32)
    return np.array([cd.std()], dtype=np.float32)


def _feat_row_wt_std(m):
    """Row weight standard deviation (spread of stabilizer weight)."""
    rw = m.sum(axis=1).astype(np.float32)
    return np.array([rw.std()], dtype=np.float32)


# --- Group B: Gaussian elimination related ---
def _feat_rank(m):
    """GF(2) rank of the binary matrix."""
    return np.array([float(_gf2_rank(m))], dtype=np.float32)


def _feat_active_qubits(m):
    """Number of columns with at least one 1."""
    return np.array([float(np.sum(np.any(m, axis=0)))], dtype=np.float32)


def _feat_active_rows(m):
    """Number of rows with at least one 1."""
    return np.array([float(np.sum(np.any(m, axis=1)))], dtype=np.float32)


def _feat_pivot_candidates(m):
    """Number of columns with degree exactly 1 (natural pivot candidates)."""
    col_deg = m.sum(axis=0)
    return np.array([float(np.sum(col_deg == 1))], dtype=np.float32)


def _feat_pivot_candidate_ratio(m):
    """pivot_candidates / active_qubits (normalised pivot availability)."""
    cd = m.sum(axis=0)
    active = np.sum(cd > 0)
    if active == 0:
        return np.array([0.0], dtype=np.float32)
    return np.array([float(np.sum(cd == 1)) / active], dtype=np.float32)


def _feat_pivot_quality(m):
    """Proportion of columns with degree <= 2 among all active columns."""
    cd = m.sum(axis=0)
    active = np.sum(cd > 0)
    if active == 0:
        return np.array([0.0], dtype=np.float32)
    return np.array([float(np.sum(cd <= 2)) / active], dtype=np.float32)


def _feat_degree_entropy(m):
    """Entropy of column degree distribution (uncertainty of pivot choices)."""
    cd = m.sum(axis=0)
    cd = cd[cd > 0]  # only active columns
    if len(cd) == 0:
        return np.array([0.0], dtype=np.float32)
    counts = np.bincount(cd.astype(int))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    ent = -float(np.sum(probs * np.log(probs)))
    return np.array([ent], dtype=np.float32)


def _feat_fill_in_estimate(m):
    """Estimated fill-in potential: sum over rows of (col_deg[row] choose 2).
    When pivoting on a row, non-pivot columns with 1s in the pivot row
    get XORed together — each pair creates fill-in. Normalised by total ones."""
    rw = m.sum(axis=1)
    # For each row: choose2(rw) = rw * (rw - 1) / 2
    fill = np.sum(rw * (rw - 1) / 2)
    total = float(m.sum())
    if total == 0:
        return np.array([0.0], dtype=np.float32)
    return np.array([float(fill) / total], dtype=np.float32)


def _feat_fill_in_raw(m):
    """Raw estimated fill-in total (unnormalised): sum of choose2(row_wt)."""
    rw = m.sum(axis=1)
    return np.array([float(np.sum(rw * (rw - 1) / 2))], dtype=np.float32)


def _feat_col_overlap_ratio(m):
    """Ratio of overlapping column pairs to total column pairs.
    Two columns overlap if they share at least one row with 1 in both."""
    ncols = m.shape[1]
    if ncols <= 1:
        return np.array([0.0], dtype=np.float32)
    total_pairs = ncols * (ncols - 1) / 2
    # Column co-occurrence: M^T * M gives overlap counts
    overlap_mat = m.T @ m  # shape (ncols, ncols)
    # Count pairs where overlap > 0
    overlap_mat_upper = np.triu(overlap_mat, k=1)
    overlapping_pairs = np.sum(overlap_mat_upper > 0)
    return np.array([float(overlapping_pairs) / total_pairs], dtype=np.float32)


# --- Group C: Structure ---
def _feat_row_sim_mean(m):
    """Mean pairwise row similarity (overlap / max weight)."""
    mu, _ = _row_similarity_pairs(m)
    return np.array([mu], dtype=np.float32)


def _feat_row_sim_max(m):
    """Max pairwise row similarity."""
    _, mx = _row_similarity_pairs(m)
    return np.array([mx], dtype=np.float32)


def _feat_col_sim_mean(m):
    """Mean pairwise column similarity."""
    mu, _ = _col_similarity_pairs(m)
    return np.array([mu], dtype=np.float32)


def _feat_col_sim_max(m):
    """Max pairwise column similarity."""
    _, mx = _col_similarity_pairs(m)
    return np.array([mx], dtype=np.float32)


# --- Group D: Quantum circuit related ---
# (col_deg, row_wt, col_deg_stats, row_wt_stats already cover:
#  qubit participation, stabilizer weight, check weight)
# Additional quantum-relevant features:

def _feat_degree_variance_ratio(m):
    """Coefficient of variation of column degrees: std/mean.
    High values indicate widely varying qubit participation."""
    cd = m.sum(axis=0).astype(np.float64)
    mean = cd.mean()
    if mean == 0:
        return np.array([0.0], dtype=np.float32)
    return np.array([float(cd.std() / mean)], dtype=np.float32)


def _feat_low_deg_ratio(m):
    """Ratio of columns with degree 1 (immediately solvable qubits)."""
    cd = m.sum(axis=0)
    active = np.sum(cd > 0)
    if active == 0:
        return np.array([0.0], dtype=np.float32)
    return np.array([float(np.sum(cd == 1)) / active], dtype=np.float32)


# Registry mapping feature names to extractor functions
FEATURE_REGISTRY = {
    # Group A: Statistics
    "flatten":          _feat_flatten,
    "col_deg":          _feat_col_deg,
    "row_wt":           _feat_row_wt,
    "density":          _feat_density,
    "sparsity":         _feat_sparsity,
    "num_ones":         _feat_num_ones,
    "col_deg_stats":    _feat_col_deg_stats,
    "row_wt_stats":     _feat_row_wt_stats,
    "col_deg_std":      _feat_col_deg_std,
    "row_wt_std":       _feat_row_wt_std,
    # Group B: Gaussian Elimination
    "rank":               _feat_rank,
    "active_qubits":      _feat_active_qubits,
    "active_rows":        _feat_active_rows,
    "pivot_candidates":   _feat_pivot_candidates,
    "pivot_candidate_ratio": _feat_pivot_candidate_ratio,
    "pivot_quality":      _feat_pivot_quality,
    "degree_entropy":     _feat_degree_entropy,
    "fill_in_estimate":   _feat_fill_in_estimate,
    "fill_in_raw":        _feat_fill_in_raw,
    "col_overlap_ratio":  _feat_col_overlap_ratio,
    # Group C: Structure
    "row_sim_mean":       _feat_row_sim_mean,
    "row_sim_max":        _feat_row_sim_max,
    "col_sim_mean":       _feat_col_sim_mean,
    "col_sim_max":        _feat_col_sim_max,
    # Group D: Quantum
    "degree_variance_ratio": _feat_degree_variance_ratio,
    "low_deg_ratio":      _feat_low_deg_ratio,
}


def compute_feature_dim(matrix_shape, feature_names):
    """Compute output dimension for a given feature set without materialising."""
    m_rows, m_cols = matrix_shape
    DIMS_MAP = {
        # Per-column/row features
        "flatten": m_rows * m_cols,
        "col_deg": m_cols,
        "row_wt":  m_rows,
        # Stats (3-dim): min, mean, max
        "col_deg_stats": 3,
        "row_wt_stats":  3,
        # Std (1-dim)
        "col_deg_std": 1,
        "row_wt_std":  1,
        # All other features are 1-dim scalars
    }
    dim = 0
    for name in feature_names:
        if name in DIMS_MAP:
            dim += DIMS_MAP[name]
        elif name in FEATURE_REGISTRY:
            dim += 1  # All unlisted features are 1-dim scalars
        else:
            raise ValueError(f"Unknown feature: {name}")
    return dim


def matrix_to_input(matrix_state, include_features=True, feature_names=None):
    """
    Encode matrix state as a flat 1-D float32 array.

    Args:
        matrix_state:  2-D binary numpy array (M×N).
        include_features:  Legacy compatibility — ignored if feature_names is given.
        feature_names:  List of feature names from FEATURE_REGISTRY.
                        Default (None) → baseline: ["flatten"].

    Returns:
        1-D float32 numpy array of concatenated features.
    """
    if feature_names is None:
        feature_names = ["flatten"]
    parts = []
    for name in feature_names:
        parts.append(FEATURE_REGISTRY[name](matrix_state))
    return np.concatenate(parts).astype(np.float32)


class SteaneValueNet(nn.Module):
    """MLP Value Network — configurable hidden dimensions."""
    def __init__(self, input_dim, hidden_dims, dropout=0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """x: (B, flat_dim) float tensor."""
        return self.net(x).squeeze(-1)


class RowColValueNet(nn.Module):
    """Row/Column Transformer Value Network.

    Replaces flatten+MLP with Row/Column Transformer encoder +
    shallow MLP value head.

    Input:  (B, R, C) binary matrix tensor.
    Output: (B,) scalar predicted cost.
    """
    def __init__(self, embed_dim=128, nhead=4, num_layers=2,
                 hidden_dims=None, max_size=200, dropout=0.1):
        super().__init__()
        self.matrix_encoder = MatrixEncoder(
            embed_dim=embed_dim,
            nhead=nhead,
            num_layers=num_layers,
            max_size=max_size,
            dropout=dropout,
        )
        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        prev = embed_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.value_head = nn.Sequential(*layers)

    def forward(self, matrix):
        """matrix: (B, R, C) int/float with values in {0,1}."""
        emb = self.matrix_encoder(matrix)
        return self.value_head(emb).squeeze(-1)


class RowColRankValueNet(nn.Module):
    """Row/Column Transformer Value Network + GF(2) rank feature.

    Identical MatrixEncoder as RowColValueNet, but concatenates
    the matrix embedding with a normalized rank scalar before the
    value head.  The rank is expected as a separate float tensor
    (B, 1), already normalised to [0, 1] by the caller.

    Input:  (matrix, rank)
        matrix:  (B, R, C) binary matrix tensor  — same as RowColValueNet
        rank:    (B, 1) float tensor in [0, 1]   — normalised GF(2) rank
    Output: (B,) scalar predicted cost.
    """
    def __init__(self, embed_dim=128, nhead=4, num_layers=2,
                 hidden_dims=None, max_size=200, dropout=0.1):
        super().__init__()
        self.matrix_encoder = MatrixEncoder(
            embed_dim=embed_dim,
            nhead=nhead,
            num_layers=num_layers,
            max_size=max_size,
            dropout=dropout,
        )
        if hidden_dims is None:
            hidden_dims = [64, 32]

        # Value head: embed_dim + 1 (rank) → output
        layers = []
        prev = embed_dim + 1
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.value_head = nn.Sequential(*layers)

    def forward(self, matrix, rank):
        """matrix: (B, R, C), rank: (B, 1) normalised [0,1]."""
        emb = self.matrix_encoder(matrix)              # (B, D)
        combined = torch.cat([emb, rank], dim=-1)      # (B, D+1)
        return self.value_head(combined).squeeze(-1)


class CostNormalizer:
    """Target normalisation (z-score)."""
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def normalize(self, t):
        return (t - self.mean) / self.std

    def denormalize(self, t_norm):
        return t_norm * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, d):
        self.mean = float(d["mean"])
        self.std = float(d["std"])
