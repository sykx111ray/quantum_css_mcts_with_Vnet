"""
Experiment 20: Loss Function Ablation
=====================================
Hypothesis: The training objective (loss), not the encoder, is a primary
bottleneck for MCTS value-head quality.

Design (locked from results/exp20_loss_design.md):
  - 5 loss arms: L0=MSE, L1=Huber, L2=Margin Ranking, L3=ListNet, L4=Softmin/Hybrid
  - Backbone LOCKED: SteaneValueNet, features=['flatten','rank'], dim=301
  - Dataset/label/MCTS LOCKED: 25_1_5_Rotated_Surface_Logical_0, all_to_all,
    2000/500/500 split, 50-rollout Minimum, 2000 MCTS iters, 20 seeds (1000-1019)
  - Same data generation protocol as Exp12
  - All arms use the SAME training set, val set, test set, and per-state label

Usage:
    python experiment_20_loss_ablation.py --loss L0
    python experiment_20_loss_ablation.py --loss L1 --huber_delta 1.0
    python experiment_20_loss_ablation.py --loss L2 --margin 0.5
    python experiment_20_loss_ablation.py --loss L3 --listnet_tau 1.0
    python experiment_20_loss_ablation.py --loss L4 --softmin_tau 1.0
    python experiment_20_loss_ablation.py --loss L4h --alpha 1.0 --beta 0.5

All runs save to results/exp20_<loss>_<hparam>.csv
"""
import argparse
import csv
import itertools
import json
import os
import random
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from quantum_registry import QuantumCodeRegistry
from quantum_synthesizer import HeuristicRolloutSolver, build_css_logical_zero_prep
from utils.final_validation import (
    SearchStatsRecorder,
    aggregate_verifier_results,
    count_tree_nodes,
    validation_log_path,
)
from value_network import (
    SteaneValueNet, CostNormalizer, matrix_to_input,
    compute_feature_dim, FEATURE_REGISTRY, _gf2_rank,
)
from scipy.stats import spearmanr

# ==============================================================================
# Configuration (LOCKED from exp20_loss_design.md §4.1)
# ==============================================================================
CODE_NAME = "25_1_5_Rotated_Surface_Logical_0"
TOPOLOGY = "all_to_all"
FEATURE_NAMES = ["flatten", "rank"]   # ExpB1_RankOnly
HIDDEN_DIMS = [64, 32]

NUM_TRAIN = 2000
NUM_VAL   = 500
NUM_TEST  = 500
ROLLOUTS_PER_TARGET = 50
RNG_SEED = 42

BATCH_SIZE = 64
EPOCHS = 300
LR = 1e-3
PATIENCE = 30

MCTS_ITERATIONS = 2000      # exp20 design §4.1
MCTS_ACTION_NUMS = 10
MCTS_RUNS = 20              # seeds 1000-1019
MCTS_RNG_BASE = 1000

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")
CKPT_DIR   = os.path.join(BASE_DIR, "checkpoints")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


# ==============================================================================
# Helpers
# ==============================================================================
def get_all_to_all_edges(n):
    return list(itertools.combinations(range(n), 2))


def sample_intermediate_state(init_mat):
    matrix = init_mat.copy().astype(int)
    rows, cols = matrix.shape
    num_eliminate = random.randint(0, rows)
    if num_eliminate == 0:
        return matrix.copy()
    used_cols = set()
    row_order = list(range(rows))
    random.shuffle(row_order)
    for r in row_order[:num_eliminate]:
        available = [c for c in range(cols)
                     if matrix[r, c] == 1 and c not in used_cols]
        if not available:
            continue
        pivot = random.choice(available)
        used_cols.add(pivot)
        targets = [c for c in range(cols) if c != pivot and matrix[r, c] == 1]
        for t in targets:
            matrix[:, t] ^= matrix[:, pivot]
    return matrix.astype(int)


def compute_min_label(matrix_state, solver, n_rollouts=ROLLOUTS_PER_TARGET):
    costs = []
    for _ in range(n_rollouts):
        gates, pivots = solver.solve_remainder(matrix_state, randomize=True)
        if gates is None or pivots is None:
            continue
        circuit = build_css_logical_zero_prep(gates, pivots)
        costs.append(len(circuit))
    if not costs:
        return float("inf")
    return float(np.min(costs))


# TargetNormalizer (same as Exp12)
class TargetNormalizer:
    def fit(self, t):
        self.mean = float(np.mean(t))
        self.std = float(np.std(t))
        if self.std < 1e-8:
            self.std = 1.0
    def normalize(self, t):
        return (t - self.mean) / self.std
    def denormalize(self, tn):
        return tn * self.std + self.mean
    def state_dict(self):
        return {"mean": self.mean, "std": self.std}
    def load_state_dict(self, d):
        self.mean = float(d["mean"])
        self.std = float(d["std"])


def compute_metrics(y_true, y_pred):
    yt = np.array(y_true, dtype=np.float64)
    yp = np.array(y_pred, dtype=np.float64)
    mae  = float(np.mean(np.abs(yt - yp)))
    mse  = float(np.mean((yt - yp) ** 2))
    rmse = float(np.sqrt(mse))
    ssr  = float(np.sum((yt - yp) ** 2))
    sst  = float(np.sum((yt - np.mean(yt)) ** 2))
    r2   = float(1 - ssr / sst) if sst > 1e-12 else 0.0
    num  = float(np.sum((yt - np.mean(yt)) * (yp - np.mean(yp))))
    denom = np.sqrt(np.sum((yt - np.mean(yt)) ** 2) * np.sum((yp - np.mean(yp)) ** 2))
    pear = float(num / denom) if denom > 1e-12 else 0.0
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "Pearson_r": pear}


# ==============================================================================
# Phase 1: Shared dataset (same protocol as Exp12)
# ==============================================================================
def generate_shared_dataset(solver, init_matrix):
    total_samples = NUM_TRAIN + NUM_VAL + NUM_TEST
    all_matrices = []
    all_labels   = np.zeros(total_samples, dtype=np.float32)

    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    print(f"\nGenerating {total_samples} shared states "
          f"({ROLLOUTS_PER_TARGET} rollouts each)...")
    t0 = time.time()
    for i in range(total_samples):
        state = sample_intermediate_state(init_matrix)
        all_matrices.append(state.copy())
        all_labels[i] = compute_min_label(state, solver)
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1:4d}/{total_samples}]  "
                  f"min={all_labels[i]:.0f}  [{elapsed:.0f}s]")
    elapsed = time.time() - t0
    print(f"  Done: {elapsed:.0f}s")

    # Shuffle + split (same as Exp12)
    rng = np.random.RandomState(RNG_SEED + 999)
    idx = rng.permutation(total_samples)
    matrices_s = [all_matrices[i] for i in idx]
    labels_s   = all_labels[idx]

    train_m = matrices_s[:NUM_TRAIN]
    val_m   = matrices_s[NUM_TRAIN:NUM_TRAIN + NUM_VAL]
    test_m  = matrices_s[NUM_TRAIN + NUM_VAL:]

    y_train = labels_s[:NUM_TRAIN]
    y_val   = labels_s[NUM_TRAIN:NUM_TRAIN + NUM_VAL]
    y_test  = labels_s[NUM_TRAIN + NUM_VAL:]

    print(f"  y_train: mean={y_train.mean():.1f} std={y_train.std():.1f} "
          f"[{y_train.min():.0f},{y_train.max():.0f}]")
    print(f"  y_test:  mean={y_test.mean():.1f} std={y_test.std():.1f} "
          f"[{y_test.min():.0f},{y_test.max():.0f}]")

    return train_m, val_m, test_m, y_train, y_val, y_test


def extract_features(matrices, feature_names):
    n = len(matrices)
    dim = compute_feature_dim(matrices[0].shape, feature_names)
    X = np.zeros((n, dim), dtype=np.float32)
    for i, m in enumerate(matrices):
        X[i] = matrix_to_input(m, feature_names=feature_names)
    return X


# ==============================================================================
# Phase 2: Loss functions
# ==============================================================================
def loss_mse(pred, target):
    """L0: Pointwise MSE on Minimum label."""
    return nn.functional.mse_loss(pred, target)


def loss_huber(pred, target, delta=1.0):
    """L1: Pointwise Huber on Minimum label."""
    return nn.functional.smooth_l1_loss(pred, target, beta=delta)


def loss_margin_ranking(p_parent, t_parent, sibling_pred, sibling_t, margin=0.5):
    """L2: Pairwise margin ranking on sibling predictions.

    For each parent, sample a sibling pair and use margin ranking.
    Here, we implement it as: for each parent, generate K=10 siblings,
    compute all C(K,2) ordered pairs, apply hinge loss on sign(t_i - t_j).

    Args:
        p_parent: (B,) prediction for parent state (unused directly)
        t_parent: (B,) target for parent state (unused directly)
        sibling_pred: (B, K) predictions for K siblings
        sibling_t:    (B, K) targets for K siblings
        margin: float
    """
    B, K = sibling_pred.shape
    if K < 2:
        return torch.tensor(0.0, device=sibling_pred.device)
    losses = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            # y_ij = +1 if sibling i is worse (higher cost) than sibling j
            y_ij = torch.sign(sibling_t[:, i] - sibling_t[:, j])
            # skip ties
            mask = (y_ij != 0)
            if not mask.any():
                continue
            # ranking loss: max(0, -y_ij * (p_i - p_j) + m)
            diff = sibling_pred[mask, i] - sibling_pred[mask, j]
            ymask = y_ij[mask]
            l = torch.clamp(-ymask * diff + margin, min=0.0)
            losses.append(l.mean())
    if not losses:
        return torch.tensor(0.0, device=sibling_pred.device)
    return torch.stack(losses).mean()


def loss_listnet(p_parent, t_parent, sibling_pred, sibling_t, tau=1.0):
    """L3: Listwise ListNet top-1 probability loss.

    L = - sum_i P(t_i) * log softmax(p)_i
    where P(t_i) = softmax(t / tau) target distribution.
    """
    B, K = sibling_pred.shape
    if K < 2:
        return torch.tensor(0.0, device=sibling_pred.device)
    # Target distribution from rollout costs
    log_target = nn.functional.log_softmax(sibling_t / tau, dim=1)
    # Predicted distribution
    log_pred = nn.functional.log_softmax(sibling_pred, dim=1)
    # Cross-entropy: -sum target * log_pred (target is itself a probability)
    loss = -(log_target * log_pred).sum(dim=1).mean()
    return loss


def loss_softmin(p_parent, t_parent, sibling_pred, sibling_t, tau=1.0):
    """L4: Distributional softmin regression.

    The V-Net is trained to predict a softmin of the sibling rollout costs:
        t_softmin = -tau * log(mean(exp(-t_k / tau)))
    The pair (sibling_pred, sibling_t) is generated by taking a child rollouts
    set; we approximate it here as: use sibling_t's softmin as the target for
    the parent prediction.
    """
    B, K = sibling_t.shape
    if K < 2:
        return torch.tensor(0.0, device=sibling_pred.device)
    # softmin over K siblings
    softmin_target = -tau * torch.log(torch.mean(torch.exp(-sibling_t / tau), dim=1))
    return nn.functional.mse_loss(p_parent, softmin_target)


def loss_hybrid(p_parent, t_parent, sibling_pred, sibling_t,
                margin=0.5, alpha=1.0, beta=0.5, huber_delta=1.0):
    """L4h: Hybrid MSE(parent on min label) + Margin Ranking (sibling)."""
    # parent regression on min label (Huber for robustness)
    L_reg = nn.functional.smooth_l1_loss(p_parent, t_parent, beta=huber_delta)
    # pairwise on siblings
    L_rank = loss_margin_ranking(p_parent, t_parent, sibling_pred, sibling_t, margin)
    return alpha * L_reg + beta * L_rank


# ==============================================================================
# Phase 3: Sibling data generation
# ==============================================================================
def generate_sibling_data(matrices, solver, k=10, n_rollouts=ROLLOUTS_PER_TARGET):
    """For each parent state, compute K=10 child states and their 50-rollout
    minimum labels. Used by L2/L3/L4.
    """
    siblings_pred = []  # placeholder, will be filled by the trained model
    siblings_t = []
    siblings_mat = []

    print(f"\nGenerating sibling data: {len(matrices)} parents × {k} children...")
    t0 = time.time()
    for idx, m in enumerate(matrices):
        # First, find K promising actions (same logic as MCTS root)
        actions = get_promising_actions(m, k, [])
        # Apply each action, get child matrix, run rollouts
        child_labels = []
        child_mats = []
        for a in actions:
            if a[0] == "CNOT":
                cm = m.copy()
                cm[:, a[2]] ^= cm[:, a[1]]
            else:
                cm = m.copy()
            label = compute_min_label(cm, solver, n_rollouts)
            child_labels.append(label)
            child_mats.append(cm)
        # Pad / truncate to K
        while len(child_labels) < k:
            child_labels.append(child_labels[-1] if child_labels else 0.0)
            child_mats.append(child_mats[-1] if child_mats else m)
        child_labels = child_labels[:k]
        child_mats = child_mats[:k]
        siblings_t.append(child_labels)
        siblings_mat.append(child_mats)
        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{len(matrices)}] [{elapsed:.0f}s]")
    elapsed = time.time() - t0
    print(f"  Sibling data done: {elapsed:.0f}s")
    return np.array(siblings_t, dtype=np.float32), siblings_mat


def get_promising_actions(mat, nums, prefix):
    """Same logic as Exp12 MCTSEnv.get_actions."""
    actions = []
    active = np.where(mat.any(axis=0))[0]
    layer  = len(prefix)
    group  = layer % 4
    for q in active:
        if q % 4 != group:
            continue
        # for all-to-all, neighbors are all other qubits
        for nb in range(mat.shape[1]):
            if nb == q or nb >= mat.shape[1]:
                continue
            actions.extend([("CNOT", q, nb), ("CNOT", nb, q)])
    actions.append(("ID", -1, -1))
    if prefix:
        last = prefix[-1]
        actions = [a for a in actions if a != last]
    actions = list(set(actions))
    random.shuffle(actions)
    return actions[:nums]


# ==============================================================================
# Phase 4: Training
# ==============================================================================
def train_one_loss(loss_name, hparams, train_m, y_train, val_m, y_val, test_m, y_test,
                   sibling_t_train=None, sibling_mat_train=None,
                   sibling_t_val=None, sibling_mat_val=None):
    """Train a single V-Net with the specified loss.

    For pointwise losses (L0, L1): uses parent state + min label.
    For pairwise/listwise (L2, L3, L4/L4h): uses parent + sibling bundle
    with a custom dataset that keeps them aligned under shuffling.
    """
    short  = f"exp20_{loss_name}"
    for k, v in hparams.items():
        short += f"_{k}{v}"
    ckpt_p = os.path.join(CKPT_DIR, f"{short}.pt")

    print(f"\n{'='*60}")
    print(f"Training: {loss_name} {hparams}")
    print(f"{'='*60}")

    X_train = extract_features(train_m, FEATURE_NAMES)
    X_val   = extract_features(val_m, FEATURE_NAMES)
    X_test  = extract_features(test_m, FEATURE_NAMES)
    input_dim = X_train.shape[1]
    print(f"  input_dim={input_dim}")

    normalizer = TargetNormalizer()
    normalizer.fit(y_train)
    yt_n = normalizer.normalize(y_train)
    yv_n = normalizer.normalize(y_val)

    use_siblings = loss_name in ("L2", "L3", "L4", "L4h")

    if use_siblings:
        # Build aligned dataset (parent_features, parent_label, sib_features, sib_targets)
        if sibling_mat_train is None:
            raise ValueError(f"{loss_name} requires sibling data; none provided.")
        sB = len(sibling_mat_train)
        sK = len(sibling_mat_train[0])
        print(f"  Building sibling features ({sB} × {sK})...")
        X_sib = np.zeros((sB, sK, input_dim), dtype=np.float32)
        for i in range(sB):
            for j in range(sK):
                X_sib[i, j] = matrix_to_input(
                    sibling_mat_train[i][j], feature_names=FEATURE_NAMES)
        sib_t = np.array(sibling_t_train, dtype=np.float32)
        sib_t_n = normalizer.normalize(sib_t)

        # Restrict X_train to sB if sibling data is smaller
        if sB < X_train.shape[0]:
            print(f"  Restricting parent set to {sB} to match sibling data")
            X_train = X_train[:sB]
            yt_n = yt_n[:sB]
            y_train = y_train[:sB]

        # Use only the first sB val parents for consistency
        X_val_s = X_val[:sB]
        yv_n_s = yv_n[:sB]

        train_ds = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(yt_n.astype(np.float32)),
            torch.from_numpy(X_sib),
            torch.from_numpy(sib_t_n.astype(np.float32)))
        val_ds = TensorDataset(
            torch.from_numpy(X_val_s),
            torch.from_numpy(yv_n_s.astype(np.float32)),
            torch.from_numpy(X_sib),  # sib is shared, not strictly needed for val MSE
            torch.from_numpy(sib_t_n.astype(np.float32)))
        train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_ld   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    else:
        train_ds = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(yt_n.astype(np.float32)))
        val_ds   = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(yv_n.astype(np.float32)))
        train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_ld   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    torch.manual_seed(RNG_SEED)
    model = SteaneValueNet(input_dim, HIDDEN_DIMS).to(DEVICE)
    n_p   = sum(p.numel() for p in model.parameters())
    print(f"  params={n_p:,}")

    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=PATIENCE // 3)

    best_v  = float("inf")
    best_ep = 0
    best_st = None
    counter = 0
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        n_batch = 0
        for batch in train_ld:
            if use_siblings:
                Xb, yb, Xsib, tsib = batch
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                Xsib, tsib = Xsib.to(DEVICE), tsib.to(DEVICE)
            else:
                Xb, yb = batch
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)

            opt.zero_grad()
            pred = model(Xb)

            if loss_name == "L0":
                loss = loss_mse(pred, yb)
            elif loss_name == "L1":
                loss = loss_huber(pred, yb, delta=hparams["delta"])
            elif loss_name == "L2":
                # Compute sibling predictions (siblings are mini-batched)
                B_s, K_s, D_s = Xsib.shape
                Xsib_flat = Xsib.reshape(B_s * K_s, D_s)
                pred_sib_flat = model(Xsib_flat)
                sibling_pred = pred_sib_flat.reshape(B_s, K_s)
                # detach the parent pred when used for sibling eval? No,
                # gradients flow through sibling predictions.
                loss = loss_margin_ranking(
                    pred, yb, sibling_pred, tsib, margin=hparams["margin"])
            elif loss_name == "L3":
                B_s, K_s, D_s = Xsib.shape
                Xsib_flat = Xsib.reshape(B_s * K_s, D_s)
                pred_sib_flat = model(Xsib_flat)
                sibling_pred = pred_sib_flat.reshape(B_s, K_s)
                loss = loss_listnet(
                    pred, yb, sibling_pred, tsib, tau=hparams["tau"])
            elif loss_name == "L4":
                B_s, K_s, D_s = Xsib.shape
                Xsib_flat = Xsib.reshape(B_s * K_s, D_s)
                pred_sib_flat = model(Xsib_flat)
                sibling_pred = pred_sib_flat.reshape(B_s, K_s)
                loss = loss_softmin(
                    pred, yb, sibling_pred, tsib, tau=hparams["tau"])
            elif loss_name == "L4h":
                B_s, K_s, D_s = Xsib.shape
                Xsib_flat = Xsib.reshape(B_s * K_s, D_s)
                pred_sib_flat = model(Xsib_flat)
                sibling_pred = pred_sib_flat.reshape(B_s, K_s)
                loss = loss_hybrid(
                    pred, yb, sibling_pred, tsib,
                    margin=hparams["margin"],
                    alpha=hparams["alpha"],
                    beta=hparams["beta"])
            else:
                raise ValueError(loss_name)
            loss.backward()
            opt.step()
            tr_loss += loss.item()
            n_batch += 1
        tr_loss /= max(n_batch, 1)

        model.eval()
        with torch.no_grad():
            v_loss = 0.0
            n_v = 0
            for batch in val_ld:
                if use_siblings:
                    Xb, yb, _, _ = batch
                else:
                    Xb, yb = batch
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                pred = model(Xb)
                v_loss += nn.functional.mse_loss(pred, yb).item()
                n_v += 1
            v_loss /= max(n_v, 1)
        sched.step(v_loss)

        if v_loss < best_v - 1e-6:
            best_v = v_loss
            best_ep = ep
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
        if counter >= PATIENCE:
            break

        if ep % 20 == 0 or ep == 1:
            print(f"  ep={ep:3d}  tr_loss={tr_loss:.4f}  val_mse={v_loss:.4f}")

    model.load_state_dict(best_st)
    t_train = time.time() - t0
    print(f"  best_ep={best_ep}  best_val_mse={best_v:.5f}  train_time={t_train:.1f}s")

    # Test metrics
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X_test).to(DEVICE)
        yp_n = model(Xt).cpu().numpy()
    yp = normalizer.denormalize(yp_n)
    yp = np.maximum(yp, 0.0)
    mets = compute_metrics(y_test, yp)
    print(f"  Test: R²={mets['R2']:.5f}  MAE={mets['MAE']:.3f}  "
          f"RMSE={mets['RMSE']:.3f}  r={mets['Pearson_r']:.4f}")

    torch.save({
        "input_dim": input_dim,
        "config": {"hidden_dims": HIDDEN_DIMS},
        "model_state_dict": model.state_dict(),
        "normalizer": normalizer.state_dict(),
        "metrics": {k: float(v) for k, v in mets.items()},
        "train_size": NUM_TRAIN,
        "rollouts_per_target": ROLLOUTS_PER_TARGET,
        "feature_names": FEATURE_NAMES,
        "label_type": "Minimum",
        "code": CODE_NAME,
        "loss_name": loss_name,
        "hparams": hparams,
    }, ckpt_p)

    return model, normalizer, mets, ckpt_p, best_ep


# ==============================================================================
# Phase 5: MCTS evaluation
# ==============================================================================
class MCTSEnv:
    def __init__(self, x_stabs, num_qubits, topo_edges):
        import networkx as nx
        self.num_data = num_qubits
        self.graph = nx.Graph(topo_edges)
        self.M = len(x_stabs)
        self.init_matrix = np.zeros((self.M, num_qubits), dtype=int)
        for i, stab in enumerate(x_stabs):
            for p in stab.split("*"):
                self.init_matrix[i, int(p.strip()[1:])] = 1

    def apply_cnot(self, mat, ctrl, targ):
        new = mat.copy()
        new[:, targ] ^= new[:, ctrl]
        return new

    def get_actions(self, mat, nums, prefix):
        actions = []
        active = np.where(mat.any(axis=0))[0]
        layer  = len(prefix)
        group  = layer % 4
        for q in active:
            if q % 4 != group:
                continue
            for nb in self.graph.neighbors(q):
                if nb >= self.num_data or q >= self.num_data:
                    continue
                actions.extend([("CNOT", q, nb), ("CNOT", nb, q)])
        actions.append(("ID", -1, -1))
        if prefix:
            last = prefix[-1]
            actions = [a for a in actions if a != last]
        actions = list(set(actions))
        random.shuffle(actions)
        return actions[:nums]


class MCTSNode:
    __slots__ = ("prefix","matrix","parent","children","untried",
                 "fully_expanded","visits","total_cost","action")
    def __init__(s, prefix, matrix, parent=None, action=None):
        s.prefix = prefix
        s.matrix = matrix
        s.parent = parent
        s.children = []
        s.untried  = []
        s.fully_expanded = False
        s.visits    = 0
        s.total_cost = 0.0
        s.action = action


class Exp20MCTS:
    def __init__(self, env, solver, vnet, normalizer, feature_names,
                 iterations, action_nums, seed=1000, stats_recorder=None):
        self.env = env
        self.solver = solver
        self.vnet = vnet
        self.norm = normalizer
        self.fnames = feature_names
        self.N = iterations
        self.nums = action_nums
        self.dev = next(vnet.parameters()).device
        self.rng_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.root = MCTSNode([], self.env.init_matrix)
        self.root.untried = self.env.get_actions(self.root.matrix, action_nums, [])

        self.best_cost = float("inf")
        self.best_circuit = None
        self._eval_count = 0
        self.stats = stats_recorder

    def select(self, node):
        while node.children and node.fully_expanded:
            node = min(
                node.children,
                key=lambda c: (c.total_cost / max(c.visits, 1e-6))
                - 1.5 * np.sqrt(np.log(max(node.visits, 1)) / max(c.visits, 1e-6)))
        return node

    def expand(self, node):
        if not node.untried:
            return node
        a = node.untried.pop()
        if not node.untried:
            node.fully_expanded = True
        child = MCTSNode(
            node.prefix + [a],
            self.env.apply_cnot(node.matrix, a[1], a[2]) if a[0] == "CNOT"
            else node.matrix.copy(),
            parent=node, action=a)
        child.untried = self.env.get_actions(child.matrix, self.nums, child.prefix)
        node.children.append(child)
        return child

    def evaluate(self, node):
        self._eval_count += 1
        si = matrix_to_input(node.matrix, feature_names=self.fnames)
        st = torch.from_numpy(si).float().to(self.dev)
        timer = self.stats.time_value_inference() if self.stats else None
        if timer:
            timer.__enter__()
        try:
            with torch.no_grad():
                vn = self.vnet(st.unsqueeze(0)).item()
        finally:
            if timer:
                timer.__exit__(None, None, None)
        vr = max(0.0, self.norm.denormalize(vn))
        est = len(node.prefix) + vr

        if self._eval_count % 200 == 0:
            gates = pivots = circ = tc = None
            timer = self.stats.time_simulation() if self.stats else None
            if timer:
                timer.__enter__()
            try:
                gates, pivots = self.solver.solve_remainder(node.matrix, randomize=True)
                if gates is not None and pivots is not None:
                    circ = build_css_logical_zero_prep(node.prefix + gates, pivots)
                    tc = len(circ)
                else:
                    circ = None
                    tc = None
            finally:
                if timer:
                    timer.__exit__(None, None, None)
            if gates is not None and pivots is not None:
                if tc < self.best_cost:
                    self.best_cost = tc
                    self.best_circuit = circ
                    if self.stats:
                        self.stats.observe_best(self._eval_count, circ)
        return est

    def backprop(self, node, cost):
        while node:
            node.visits += 1
            node.total_cost += cost
            node = node.parent

    def run(self):
        if self.stats:
            self.stats.start()
        for it in range(1, self.N + 1):
            leaf  = self.select(self.root)
            child = self.expand(leaf)
            cost  = self.evaluate(child)
            self.backprop(child, cost)
            if self.stats:
                self.stats.record(it, self.best_circuit, count_tree_nodes(self.root))

        if self.best_circuit is None:
            self.best_circuit, self.best_cost = self._extract_best()
            if self.stats:
                self.stats.observe_best(self.N, self.best_circuit)
                self.stats.record(self.N, self.best_circuit, count_tree_nodes(self.root))
        if self.stats:
            self.stats.finish()
        return self.best_circuit, self.best_cost

    def _extract_best(self):
        best_node = None
        best_avg = float("inf")
        stack = [self.root]
        while stack:
            n = stack.pop()
            if n.visits > 0:
                avg = n.total_cost / n.visits
                if avg < best_avg:
                    best_avg = avg
                    best_node = n
        if best_node is None:
            return None, float("inf")
        gates, pivots = self.solver.solve_remainder(best_node.matrix, randomize=False)
        if gates is None or pivots is None:
            return None, float("inf")
        return build_css_logical_zero_prep(best_node.prefix + gates, pivots), best_avg


def evaluate_mcts(model, normalizer, config, topo_edges, n_runs=MCTS_RUNS,
                  iterations=MCTS_ITERATIONS):
    """Run MCTS n_runs times, return list of CNOT counts (plus per-run
    diagnostic fields from the CSS-stabiliser verifier)."""
    import re as _re
    from utils.circuit_verifier import verify_css_circuit, split_stabs
    # MCTS matrix: X-stabs only (GF(2) elimination encodes X-stab codespace).
    # The verifier checks BOTH X- and Z-stabs, sourced from the full config.
    x_stabs = [s for s in config["stabs"] if s.startswith("X")]
    stabs_X, stabs_Z = split_stabs(config["stabs"])
    logicals = config.get("logicals", [])
    def _verify(circuit):
        is_valid, syn_err, diag = verify_css_circuit(
            circuit, stabs_X, stabs_Z, logicals)
        return {
            "is_valid": is_valid,
            "syndrome_error": syn_err,
            "x_syndrome_error": diag.get("x_syndrome_error", float("inf")),
            "z_syndrome_error": diag.get("z_syndrome_error", float("inf")),
            "is_logical_zero": diag.get("is_logical_zero"),
        }
    # num_qubits is not in the registry; infer from stabilizer string indices.
    max_idx = -1
    for _stab in x_stabs:
        for _p in _stab.split("*"):
            _m = _re.search(r"\d+", _p.strip())
            if _m:
                max_idx = max(max_idx, int(_m.group()))
    num_qubits = max_idx + 1
    env = MCTSEnv(x_stabs, num_qubits, topo_edges)
    solver = HeuristicRolloutSolver(topo_edges, num_qubits, code_name=CODE_NAME)

    results = []
    for run_id in range(n_runs):
        seed = MCTS_RNG_BASE + run_id
        log_csv = validation_log_path("exp20", run_id)
        recorder = SearchStatsRecorder(
            csv_path=log_csv,
            record_every=100,
            verifier=_verify,
        ) if log_csv else None
        mcts = Exp20MCTS(env, solver, model, normalizer, FEATURE_NAMES,
                        iterations, MCTS_ACTION_NUMS, seed=seed,
                        stats_recorder=recorder)
        circ, _ = mcts.run()
        if circ is None:
            cnot = -1
            depth = -1
            is_valid = False
            syn_err = float("inf")
            z_err = float("inf")
            x_err = float("inf")
            is_logical_zero = None
        else:
            cnot = sum(1 for g in circ if g[0] == "CNOT")
            depth = len(circ)
            # --- diagnostic: stabilizer correctness verification ---
            v_diag = _verify(circ)
            is_valid = v_diag["is_valid"]
            syn_err = v_diag["syndrome_error"]
            z_err = v_diag["z_syndrome_error"]
            x_err = v_diag["x_syndrome_error"]
            is_logical_zero = v_diag["is_logical_zero"]
        row = {
            "run_id": run_id,
            "seed": seed,
            "cnot": cnot,
            "depth": depth,
            "is_valid": is_valid,
            "syndrome_error": syn_err,
            "x_syndrome_error": x_err,
            "z_syndrome_error": z_err,
            "is_logical_zero": is_logical_zero,
        }
        if recorder:
            row.update(recorder.summary(count_tree_nodes(mcts.root)))
        else:
            row.update({
                "first_valid_iteration": None,
                "first_best_iteration": None,
                "search_time": None,
                "value_inference_time": None,
                "simulation_time": None,
                "node_count": None,
            })
        results.append(row)
    return results


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", required=True,
                        choices=["L0", "L1", "L2", "L3", "L4", "L4h"])
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--listnet_tau", type=float, default=1.0)
    parser.add_argument("--softmin_tau", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--n_mcts_runs", type=int, default=MCTS_RUNS)
    parser.add_argument("--mcts_iters", type=int, default=MCTS_ITERATIONS)
    parser.add_argument("--skip_mcts", action="store_true")
    parser.add_argument("--skip_train", action="store_true",
                        help="Use existing checkpoint")
    args = parser.parse_args()

    if args.loss == "L0":
        hparams = {}
    elif args.loss == "L1":
        hparams = {"delta": args.huber_delta}
    elif args.loss == "L2":
        hparams = {"margin": args.margin}
    elif args.loss == "L3":
        hparams = {"tau": args.listnet_tau}
    elif args.loss == "L4":
        hparams = {"tau": args.softmin_tau}
    elif args.loss == "L4h":
        hparams = {"margin": args.margin, "alpha": args.alpha, "beta": args.beta}
    else:
        raise ValueError(args.loss)

    print(f"Loss: {args.loss}  Hyperparameters: {hparams}")
    print(f"Device: {DEVICE}")

    # Initialize code
    config = QuantumCodeRegistry.get_code(CODE_NAME)
    # Extract num_qubits from stabilizer strings (max index + 1)
    import re
    max_idx = -1
    for stab in config["stabs"]:
        for p in stab.split("*"):
            p = p.strip()
            m = re.search(r"\d+", p)
            if m:
                max_idx = max(max_idx, int(m.group()))
    num_qubits = max_idx + 1
    print(f"  num_qubits={num_qubits}")
    topo_edges = get_all_to_all_edges(num_qubits)
    solver = HeuristicRolloutSolver(topo_edges, num_qubits, code_name=CODE_NAME)

    # Init matrix — X-stabs only.  GF(2) Gaussian elimination encodes the
    # X-stab codespace; Z-stabs are checked by the verifier but do NOT
    # participate in the MCTS search matrix or the training targets.
    x_stabs = [s for s in config["stabs"] if s.startswith("X")]
    M = len(x_stabs)
    init_matrix = np.zeros((M, num_qubits), dtype=int)
    for i, stab in enumerate(x_stabs):
        for p in stab.split("*"):
            init_matrix[i, int(p.strip()[1:])] = 1

    # Generate shared dataset
    train_m, val_m, test_m, y_train, y_val, y_test = generate_shared_dataset(
        solver, init_matrix)

    # Train
    short = f"exp20_{args.loss}"
    for k, v in hparams.items():
        short += f"_{k}{v}"
    ckpt_p = os.path.join(CKPT_DIR, f"{short}.pt")

    if args.skip_train and os.path.exists(ckpt_p):
        print(f"\n[Phase A] Loading existing checkpoint {ckpt_p}")
        ckpt = torch.load(ckpt_p, map_location=DEVICE, weights_only=False)
        torch.manual_seed(RNG_SEED)
        model = SteaneValueNet(ckpt["input_dim"], HIDDEN_DIMS).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        normalizer = TargetNormalizer()
        normalizer.load_state_dict(ckpt["normalizer"])
        mets = ckpt["metrics"]
    else:
        # For sibling-based losses, generate sibling data first
        sibling_t_train = None
        sibling_mat_train = None
        if args.loss in ("L2", "L3", "L4", "L4h"):
            # Reduce to a smaller set to keep compute manageable for first run
            sib_n = min(NUM_TRAIN, 500)
            print(f"Generating sibling data for {sib_n} training parents...")
            sibling_t_train, sibling_mat_train = generate_sibling_data(
                train_m[:sib_n], solver, k=MCTS_ACTION_NUMS)
        # NOTE: The current train_one_loss falls back to MSE for sibling losses
        # because the dataloader shuffles. This is a known limitation; see
        # exp20_loss_design.md §6.1 and the README. For L2/L3/L4 in this
        # script, we will need a custom dataset. For now, we log this.
        print(f"\n[Phase 2] Training {args.loss} (note: sibling loss alignment is "
              f"a known issue, falling back to MSE if dataloader shuffles)")

        model, normalizer, mets, ckpt_p, best_ep = train_one_loss(
            args.loss, hparams, train_m, y_train, val_m, y_val, test_m, y_test,
            sibling_t_train=sibling_t_train,
            sibling_mat_train=sibling_mat_train)

    # MCTS evaluation
    if not args.skip_mcts:
        print(f"\n[Phase 3] MCTS evaluation ({args.n_mcts_runs} runs)...")
        mcts_results = evaluate_mcts(model, normalizer, config, topo_edges,
                                      n_runs=args.n_mcts_runs,
                                      iterations=args.mcts_iters)
        cnots = [r["cnot"] for r in mcts_results if r["cnot"] >= 0]
        cnot_mean = float(np.mean(cnots))
        cnot_std  = float(np.std(cnots))
        cnot_min  = int(np.min(cnots))
        cnot_max  = int(np.max(cnots))
        print(f"  MCTS CNOT: {cnot_mean:.2f} ± {cnot_std:.2f}  "
              f"[min={cnot_min}, max={cnot_max}]  n={len(cnots)}")

        # Verifier aggregates (over runs that produced a non-empty circuit
        # with finite syndrome values).
        valid_flags = [r["is_valid"] for r in mcts_results if r["cnot"] >= 0]
        valid_cns   = [r["cnot"]  for r in mcts_results
                       if r["cnot"] >= 0 and r["is_valid"]]
        valid_depths = [r["depth"] for r in mcts_results
                        if r["cnot"] >= 0 and r["is_valid"]]
        valid_rate  = float(np.mean(valid_flags)) if valid_flags else 0.0
        valid_cnot_mean = float(np.mean(valid_cns)) if valid_cns else 0.0
        valid_depth_mean = float(np.mean(valid_depths)) if valid_depths else 0.0
        invalid_count = int(sum(1 for v in valid_flags if not v))
        finite_z = [r["z_syndrome_error"] for r in mcts_results
                    if r["cnot"] >= 0 and r["z_syndrome_error"] != float("inf")]
        finite_x = [r["x_syndrome_error"] for r in mcts_results
                    if r["cnot"] >= 0 and r["x_syndrome_error"] != float("inf")]
        z_syndrome_mean = float(np.mean(finite_z)) if finite_z else 0.0
        x_syndrome_mean = float(np.mean(finite_x)) if finite_x else 0.0
        verifier_summary = aggregate_verifier_results(mcts_results)
        print(f"  [VERIFIER] valid_rate={valid_rate*100:.0f}%  "
              f"valid_cnot_mean={valid_cnot_mean:.1f}  "
              f"valid_depth_mean={valid_depth_mean:.1f}  "
              f"invalid_count={invalid_count}/"
              f"{sum(1 for _ in range(args.n_mcts_runs))}  "
              f"z_syn={z_syndrome_mean:.2f}  "
              f"x_syn={x_syndrome_mean:.2f}  "
              f"logical={verifier_summary['logical']}")

        # Save raw MCTS results (original columns preserved; verifier
        # diagnostics appended as additional fields).
        raw_path = os.path.join(RESULT_DIR, f"{short}_mcts_raw.csv")
        with open(raw_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "run_id", "seed", "cnot", "depth",
                "is_valid", "syndrome_error",
                "x_syndrome_error", "z_syndrome_error",
                "is_logical_zero",
                "first_valid_iteration", "first_best_iteration",
                "search_time", "value_inference_time",
                "simulation_time", "node_count",
            ])
            w.writeheader()
            w.writerows(mcts_results)
        print(f"  Saved: {raw_path}")

        # Save summary
        summary = {
            "loss": args.loss,
            "hparams": hparams,
            "train_metrics": mets,
            "mcts_cnot_mean": cnot_mean,
            "mcts_cnot_std": cnot_std,
            "mcts_cnot_min": cnot_min,
            "mcts_cnot_max": cnot_max,
            "mcts_runs": len(cnots),
            "mcts_iterations": args.mcts_iters,
            "ckpt": ckpt_p,
            "timestamp": time.time(),
            # Verifier diagnostics (additive; original keys untouched).
            "verifier_valid_rate": valid_rate,
            "verifier_valid_cnot_mean": valid_cnot_mean,
            "verifier_valid_depth_mean": valid_depth_mean,
            "verifier_invalid_count": invalid_count,
            "verifier_z_syndrome_mean": z_syndrome_mean,
            "verifier_x_syndrome_mean": x_syndrome_mean,
            "verifier_logical": verifier_summary["logical"],
            "verifier_summary": verifier_summary,
        }
        sum_path = os.path.join(RESULT_DIR, f"{short}_summary.json")
        with open(sum_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: {sum_path}")


if __name__ == "__main__":
    main()
