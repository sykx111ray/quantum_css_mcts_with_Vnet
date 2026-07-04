"""
experiment_17_rowcol_rank.py — RowCol Transformer + GF(2) rank feature

Trains RowColRankValueNet (MatrixEncoder + rank concatenation + MLP value head)
on Surface d=5, compares against existing baselines.

Protocol identical to experiment_14:
  - Same dataset generation (shared seed 42)
  - Same Minimum label (50-rollout)
  - Same training hyperparameters
  - Same MCTS evaluation protocol

Key difference from experiment_14:
  - Model: RowColRankValueNet (matrix + rank) instead of RowColValueNet (matrix only)
  - Checkpoint: model_type = "rowcol_rank"
  - Training: passes (matrix, rank) tuple to model
  - MCTS: computes GF(2) rank at each node, passes to V-Net
"""
import csv
import itertools
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr

from quantum_registry import QuantumCodeRegistry
from quantum_synthesizer import HeuristicRolloutSolver, build_css_logical_zero_prep
from utils.circuit_verifier import verify_css_circuit, split_stabs
from utils.final_validation import (
    SearchStatsRecorder,
    aggregate_verifier_results,
    count_tree_nodes,
    validation_log_path,
)
from value_network import RowColRankValueNet, _gf2_rank, matrix_to_input, compute_feature_dim

# ==============================================================================
# Configuration — IDENTICAL to experiment_14
# ==============================================================================
CODE_NAME = "25_1_5_Rotated_Surface_Logical_0"
TOPOLOGY  = "all_to_all"

NUM_TRAIN  = 2000
NUM_VAL    = 500
NUM_TEST   = 500
ROLLOUTS   = 50
RNG_SEED   = 42

BATCH_SIZE  = 64
EPOCHS      = 300
LR          = 1e-3
PATIENCE    = 30
HIDDEN_DIMS = [64, 32]

# Matrix Encoder config — IDENTICAL to experiment_14
EMBED_DIM   = 128
NHEAD       = 4
NUM_LAYERS  = 2

MCTS_ITER  = 1500
MCTS_ACTS  = 10
MCTS_RUNS  = 5
MCTS_SEED  = 1000

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")
CKPT_DIR   = os.path.join(BASE_DIR, "checkpoints")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# Max rank = num_rows (12 for Surface d=5). Used for normalisation.
# Derived dynamically from init_m shape, but 12 is the known value.
MAX_RANK = None  # set in main() from init_m.shape[0]


# ==============================================================================
# Helpers — IDENTICAL to experiment_14
# ==============================================================================
def get_all_to_all_edges(n):
    return list(itertools.combinations(range(n), 2))

def sample_intermediate_state(init_mat):
    matrix = init_mat.copy().astype(int)
    rows, cols = matrix.shape
    n = random.randint(0, rows)
    if n == 0:
        return matrix.copy()
    used = set()
    order = list(range(rows))
    random.shuffle(order)
    for r in order[:n]:
        avail = [c for c in range(cols) if matrix[r,c]==1 and c not in used]
        if not avail:
            continue
        p = random.choice(avail)
        used.add(p)
        for t in range(cols):
            if t != p and matrix[r,t] == 1:
                matrix[:,t] ^= matrix[:,p]
    return matrix.astype(int)

def compute_min_label(m, solver):
    costs = []
    for _ in range(ROLLOUTS):
        g, piv = solver.solve_remainder(m, randomize=True)
        if g is not None and piv is not None:
            costs.append(len(build_css_logical_zero_prep(g, piv)))
    return float(np.min(costs)) if costs else float("inf")

def get_cnot(c):
    if c is None:
        return 0
    return sum(1 for g in c if g[0]=="CNOT")

def get_depth(c):
    if c is None:
        return 0
    return len(c)


class TargetNorm:
    def __init__(self):
        self.mu = 0.0; self.s = 1.0
    def fit(self, t):
        self.mu = float(np.mean(t)); self.s = float(np.std(t))
        if self.s < 1e-8: self.s = 1.0
    def norm(self, t):   return (t - self.mu) / self.s
    def denorm(self, t): return t * self.s + self.mu


def compute_metrics(yt, yp):
    yt = np.array(yt, dtype=np.float64); yp = np.array(yp, dtype=np.float64)
    mae = float(np.mean(np.abs(yt-yp)))
    mse = float(np.mean((yt-yp)**2))
    rmse = float(np.sqrt(mse))
    ssr = float(np.sum((yt-yp)**2))
    sst = float(np.sum((yt-np.mean(yt))**2))
    r2 = float(1-ssr/sst) if sst>1e-12 else 0.0
    num = float(np.sum((yt-np.mean(yt))*(yp-np.mean(yp))))
    denom = np.sqrt(np.sum((yt-np.mean(yt))**2)*np.sum((yp-np.mean(yp))**2))
    pr = float(num/denom) if denom>1e-12 else 0.0
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "Pearson_r": pr}


# ==============================================================================
# Dataset generation — adds rank computation for each matrix
# ==============================================================================
def generate_shared_dataset(solver, init_m):
    N = NUM_TRAIN + NUM_VAL + NUM_TEST
    mats = []
    ranks = np.zeros(N, dtype=np.float32)
    labs = np.zeros(N, dtype=np.float32)
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    print(f"Generating {N} states...")
    t0 = time.time()
    for i in range(N):
        s = sample_intermediate_state(init_m)
        mats.append(s.copy().astype(np.float32))
        ranks[i] = float(_gf2_rank(s))
        labs[i] = compute_min_label(s, solver)
        if (i+1) % 500 == 0:
            print(f"  [{i+1:4d}/{N}]  min={labs[i]:.0f}  rank={ranks[i]:.0f}  "
                  f"[{time.time()-t0:.0f}s]")
    print(f"  Done: {time.time()-t0:.0f}s")
    rng = np.random.RandomState(RNG_SEED+999)
    idx = rng.permutation(N)
    ms  = [mats[i] for i in idx]
    rks = ranks[idx]
    ls  = labs[idx]
    return (ms[:NUM_TRAIN], ms[NUM_TRAIN:NUM_TRAIN+NUM_VAL],
            ms[NUM_TRAIN+NUM_VAL:],
            rks[:NUM_TRAIN], rks[NUM_TRAIN:NUM_TRAIN+NUM_VAL],
            rks[NUM_TRAIN+NUM_VAL:],
            ls[:NUM_TRAIN], ls[NUM_TRAIN:NUM_TRAIN+NUM_VAL],
            ls[NUM_TRAIN+NUM_VAL:])


# ==============================================================================
# Training — adapted for RowColRankValueNet
# ==============================================================================
class RankDataset2D(Dataset):
    """Dataset returning (matrix, rank, label) tuple."""
    def __init__(self, matrices, ranks, targets):
        self.matrices = matrices
        self.ranks = ranks
        self.targets = targets
    def __len__(self):
        return len(self.matrices)
    def __getitem__(self, i):
        return self.matrices[i], self.ranks[i], self.targets[i]


def collate_2d_rank(batch):
    """Collate for (matrix, rank, target) batches."""
    matrices = torch.stack([item[0] for item in batch])
    ranks = torch.tensor([item[1] for item in batch], dtype=torch.float32).unsqueeze(-1)
    targets = torch.tensor([float(item[2]) for item in batch])
    return (matrices, ranks), targets


def train_model(model, train_mat, train_rank, y_train,
                val_mat, val_rank, y_val,
                test_mat, test_rank, y_test, model_type="rowcol_rank"):
    """Train RowColRankValueNet.

    model_type "rowcol_rank" passes (matrix, rank) to model.forward().
    """
    norm = TargetNorm(); norm.fit(y_train)
    yt_n = norm.norm(y_train); yv_n = norm.norm(y_val)
    train_targets = yt_n.astype(np.float32)
    val_targets   = yv_n.astype(np.float32)

    # Build 2-D datasets with rank
    train_mat_t = [torch.from_numpy(m) for m in train_mat]
    val_mat_t   = [torch.from_numpy(m) for m in val_mat]
    test_mat_t  = [torch.from_numpy(m) for m in test_mat]

    train_ds = RankDataset2D(train_mat_t, train_rank, train_targets)
    val_ds   = RankDataset2D(val_mat_t, val_rank, val_targets)

    train_ld = DataLoader(train_ds, BATCH_SIZE, True, collate_fn=collate_2d_rank)
    val_ld   = DataLoader(val_ds, BATCH_SIZE, False, collate_fn=collate_2d_rank)

    n_p = sum(p.numel() for p in model.parameters())
    print(f"  params={n_p:,}")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=PATIENCE//3)
    loss_fn = nn.MSELoss()

    best_v, best_ep, best_st, cnt = float("inf"), 0, None, 0
    for ep in range(1, EPOCHS+1):
        model.train(); tr_l = 0.0
        for (Xb, Rb), yb in train_ld:
            Xb, Rb, yb = Xb.to(DEVICE), Rb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            l = loss_fn(model(Xb, Rb), yb); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tr_l += l.item()*len(yb)
        tr_l /= NUM_TRAIN

        model.eval(); vl_l = 0.0
        with torch.no_grad():
            for (Xb, Rb), yb in val_ld:
                Xb, Rb, yb = Xb.to(DEVICE), Rb.to(DEVICE), yb.to(DEVICE)
                vl_l += loss_fn(model(Xb, Rb), yb).item()*len(yb)
        vl_l /= NUM_VAL; sched.step(vl_l)

        if vl_l < best_v:
            best_v = vl_l; best_ep = ep; cnt = 0
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            cnt += 1
        if cnt >= PATIENCE: break

    model.load_state_dict(best_st)

    # Evaluate on test set
    model.eval()
    all_preds = []
    bs = BATCH_SIZE
    for i in range(0, len(test_mat_t), bs):
        batch_m = torch.stack(test_mat_t[i:i+bs]).to(DEVICE)
        batch_r = torch.tensor(test_rank[i:i+bs], dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        with torch.no_grad():
            all_preds.append(model(batch_m, batch_r).cpu().numpy())
    yp_n = np.concatenate(all_preds)
    yp = norm.denorm(yp_n); yp = np.maximum(yp, 0.0)
    mets = compute_metrics(y_test, yp)
    print(f"  ep={best_ep:3d}  R²={mets['R2']:.5f}  MAE={mets['MAE']:.2f}  "
          f"r={mets['Pearson_r']:.4f}")
    return model, norm, mets


# ==============================================================================
# MCTS — self-contained, identical protocol to experiment_14 except model call
# ==============================================================================
class Env2D:
    def __init__(self, xs, nq, topo):
        import networkx as nx
        self.n = nq; self.g = nx.Graph(topo); self.M = len(xs)
        self.im = np.zeros((self.M, nq), dtype=int)
        for i, s in enumerate(xs):
            for p in s.split("*"): self.im[i, int(p.strip()[1:])] = 1
    def cnot(self, m, c, t):
        m2 = m.copy(); m2[:, t] ^= m2[:, c]; return m2
    def acts(self, m, pf):
        a = []; active = np.where(m.any(0))[0]; layer = len(pf)
        for q in active:
            if q % 4 != layer % 4: continue
            for nb in self.g.neighbors(q):
                if nb >= self.n or q >= self.n: continue
                a.extend([("CNOT", q, nb), ("CNOT", nb, q)])
        a.append(("ID", -1, -1))
        if pf: a = [x for x in a if x != pf[-1]]
        a = list(set(a)); random.shuffle(a); return a[:MCTS_ACTS]


class Nd2D:
    __slots__ = ("pf", "m", "p", "ch", "ut", "fe", "v", "tc", "ac")
    def __init__(s, pf, m, p=None, ac=None):
        s.pf = pf; s.m = m; s.p = p; s.ac = ac
        s.ch = []; s.ut = []; s.fe = False; s.v = 0; s.tc = 0.0


class MCTS2D:
    """MCTS using RowColRankValueNet (model_type="rowcol_rank").

    The .ev() method computes GF(2) rank from the node's matrix and passes
    a normalised rank tensor alongside the matrix tensor to the model.
    """
    def __init__(s, env, slv, vn, norm, max_rank, stats_recorder=None):
        s.e = env; s.s = slv; s.vn = vn; s.nm = norm
        s.max_r = float(max_rank)
        s.dev = next(vn.parameters()).device
        s.rt = Nd2D([], s.e.im)
        s.rt.ut = s.e.acts(s.rt.m, [])
        s.bc = float("inf"); s.br = None; s.ec = 0
        s.stats = stats_recorder
    def sel(s, n):
        while n.ch and n.fe:
            n = min(n.ch, key=lambda c: (c.tc/max(c.v, 1e-6))
                     -1.5*np.sqrt(np.log(max(n.v, 1))/max(c.v, 1e-6)))
        return n
    def exp(s, n):
        if not n.ut: return n
        a = n.ut.pop()
        if not n.ut: n.fe = True
        c = Nd2D(n.pf+[a],
                  s.e.cnot(n.m, a[1], a[2]) if a[0]=="CNOT" else n.m.copy(),
                  n, a)
        c.ut = s.e.acts(c.m, c.pf); n.ch.append(c); return c
    def ev(s, n):
        s.ec += 1
        # Matrix tensor
        st = torch.from_numpy(n.m.astype(np.float32)).to(s.dev).unsqueeze(0)
        # GF(2) rank tensor, normalised
        rk = float(_gf2_rank(n.m)) / s.max_r
        rt = torch.tensor([[rk]], dtype=torch.float32, device=s.dev)
        timer = s.stats.time_value_inference() if s.stats else None
        if timer: timer.__enter__()
        try:
            with torch.no_grad():
                vn = s.vn(st, rt).item()
        finally:
            if timer: timer.__exit__(None, None, None)
        vr = max(0.0, s.nm.denorm(vn)); est = len(n.pf) + vr
        if s.ec % 200 == 0:
            g = pv = c = tc = None
            timer = s.stats.time_simulation() if s.stats else None
            if timer: timer.__enter__()
            try:
                g, pv = s.s.solve_remainder(n.m, randomize=True)
                if g is not None and pv is not None:
                    c = build_css_logical_zero_prep(n.pf+g, pv); tc = len(c)
                else:
                    c = None; tc = None
            finally:
                if timer: timer.__exit__(None, None, None)
            if g is not None and pv is not None:
                if tc < s.bc:
                    s.bc = tc; s.br = c
                    if s.stats: s.stats.observe_best(s.ec, c)
        return est
    def bp(s, n, c):
        while n: n.v += 1; n.tc += c; n = n.p
    def run(s):
        if s.stats: s.stats.start()
        for _ in range(1, MCTS_ITER+1):
            lf = s.sel(s.rt); ch = s.exp(lf); co = s.ev(ch); s.bp(ch, co)
            if s.stats: s.stats.record(_, s.br, count_tree_nodes(s.rt))
        if s.br is None:
            s.br, s.bc = s._ext()
            if s.stats:
                s.stats.observe_best(MCTS_ITER, s.br)
                s.stats.record(MCTS_ITER, s.br, count_tree_nodes(s.rt))
        if s.stats: s.stats.finish()
        return s.br
    def _ext(s):
        bn = None; ba = float("inf"); stk = [s.rt]
        while stk:
            n = stk.pop()
            if n.v > 0:
                a = n.tc/n.v
                if a < ba: ba = a; bn = n
            stk.extend(n.ch)
        if bn is None: return None, float("inf")
        g, pv = s.s.solve_remainder(bn.m, randomize=True)
        if g is None or pv is None: return None, float("inf")
        c = build_css_logical_zero_prep(bn.pf+g, pv); return c, len(c)


def eval_mcts(ename, model, norm, max_rank, solver, im, nq, topo):
    print(f"  MCTS: {ename}  ", end="", flush=True)
    model.eval()
    results = []
    # Pre-fetch CSS stabiliser / logical-Z strings for the diagnostic
    # verifier (additive layer; does NOT change circuit or cost).
    cfg = QuantumCodeRegistry.get_code(CODE_NAME)
    stabs_X, stabs_Z = split_stabs(cfg["stabs"])
    logicals = cfg.get("logicals", [])
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
    for rid in range(MCTS_RUNS):
        seed = MCTS_SEED + rid
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        log_csv = validation_log_path("exp17", rid)
        recorder = SearchStatsRecorder(log_csv, 100, _verify) if log_csv else None
        mcts = MCTS2D(
            env=Env2D(
                xs=[s for s in QuantumCodeRegistry.get_code(CODE_NAME)["stabs"]
                    if s.startswith("X")], nq=nq, topo=topo),
            slv=solver, vn=model, norm=norm, max_rank=max_rank,
            stats_recorder=recorder)
        t0 = time.perf_counter(); best = mcts.run(); rt = time.perf_counter()-t0
        cn = get_cnot(best); dp = get_depth(best)
        ok = cn > 0
        # --- diagnostic: stabilizer correctness verification ---
        if ok:
            v_diag = _verify(best)
            is_valid = v_diag["is_valid"]
            syn_err = v_diag["syndrome_error"]
        else:
            is_valid, syn_err, v_diag = False, float("inf"), {
                "x_syndrome_error": float("inf"),
                "z_syndrome_error": float("inf"),
                "is_logical_zero": None,
                "num_cnot": 0, "num_h": 0,
            }
        stats = recorder.summary(count_tree_nodes(mcts.rt)) if recorder else {}
        results.append((cn, dp, rt, is_valid, syn_err, v_diag, stats))
        print(f"r{rid+1}={cn}{'V' if is_valid else 'I'}", end=" ", flush=True)
    cnots = [r[0] for r in results if r[0] > 0]
    depths = [r[1] for r in results if r[0] > 0]
    cn_m = float(np.mean(cnots)) if cnots else 0.0
    cn_s = float(np.std(cnots)) if len(cnots) > 1 else 0.0
    dp_m = float(np.mean(depths)) if depths else 0.0
    dp_s = float(np.std(depths)) if len(depths) > 1 else 0.0
    # Verifier aggregates (over runs that produced a non-empty circuit)
    valid_flags = [r[3] for r in results if r[0] > 0]
    valid_cns = [r[0] for r in results if r[0] > 0 and r[3]]
    valid_dns = [r[1] for r in results if r[0] > 0 and r[3]]
    valid_rate = float(np.mean(valid_flags)) if valid_flags else 0.0
    valid_cnot_mean = float(np.mean(valid_cns)) if valid_cns else 0.0
    valid_depth_mean = float(np.mean(valid_dns)) if valid_dns else 0.0
    invalid_count = int(sum(1 for r in results if r[0] > 0 and not r[3]))
    finite_z = [r[5].get("z_syndrome_error", float("inf")) for r in results
                if r[0] > 0 and r[5].get("z_syndrome_error", float("inf")) != float("inf")]
    finite_x = [r[5].get("x_syndrome_error", float("inf")) for r in results
                if r[0] > 0 and r[5].get("x_syndrome_error", float("inf")) != float("inf")]
    z_err_mean = float(np.mean(finite_z)) if finite_z else 0.0
    x_err_mean = float(np.mean(finite_x)) if finite_x else 0.0
    per_run = []
    for rid, row in enumerate(results):
        diag = row[5]
        per_run.append({
            "run_id": rid,
            "cnot": row[0],
            "depth": row[1],
            "time": row[2],
            "is_valid": row[3],
            "syndrome_error": row[4],
            "x_syndrome_error": diag.get("x_syndrome_error", float("inf")),
            "z_syndrome_error": diag.get("z_syndrome_error", float("inf")),
            "is_logical_zero": diag.get("is_logical_zero"),
            **row[6],
        })
    verifier_summary = aggregate_verifier_results(per_run)
    print(f"→ {cn_m:.1f}±{cn_s:.1f}  [valid={valid_rate*100:.0f}%]")
    return {"ename": ename, "cnot_mean": cn_m, "cnot_std": cn_s,
            "depth_mean": dp_m, "depth_std": dp_s,
            "valid_rate": valid_rate, "valid_cnot_mean": valid_cnot_mean,
            "valid_depth_mean": valid_depth_mean,
            "invalid_count": invalid_count, "z_syndrome_mean": z_err_mean,
            "x_syndrome_mean": x_err_mean,
            "logical": verifier_summary["logical"],
            "verifier_summary": verifier_summary,
            "per_run": per_run}


# ==============================================================================
# Main
# ==============================================================================
def main():
    global MAX_RANK

    print("=" * 60)
    print("Experiment 17: RowCol Transformer + Rank")
    print("=" * 60)
    print(f"Code: {CODE_NAME}, Label: Minimum ({ROLLOUTS}-rollout)")
    print(f"Device: {DEVICE}")

    # Setup
    cfg = QuantumCodeRegistry.get_code(CODE_NAME)
    xs = [s for s in cfg["stabs"] if s.startswith("X")]
    # Derive num_qubits from stabiliser strings (works for any code size)
    import re
    max_idx = -1
    for stab in cfg["stabs"]:
        for p in stab.split("*"):
            m = re.search(r"\d+", p.strip())
            if m: max_idx = max(max_idx, int(m.group()))
    nq = max_idx + 1
    init_m = np.zeros((len(xs), nq), dtype=int)
    for i, s in enumerate(xs):
        for p in s.split("*"):
            init_m[i, int(p.strip()[1:])] = 1
    MAX_RANK = float(init_m.shape[0])  # 12 for Surface d=5, 40 for d=9
    topo = get_all_to_all_edges(nq)
    solver = HeuristicRolloutSolver(topo, nq, code_name=CODE_NAME)

    # Generate shared dataset (includes rank)
    print()
    tM, vM, tsM, tR, vR, tsR, yt, yv, yts = generate_shared_dataset(solver, init_m)
    print(f"  y_train: mean={yt.mean():.1f} std={yt.std():.1f}  "
          f"rank_train: mean={tR.mean():.1f} std={tR.std():.1f}")

    # Train
    print(f"\n{'='*60}")
    print(f"Training: RowCol + Rank")
    print(f"{'='*60}")
    model = RowColRankValueNet(
        embed_dim=EMBED_DIM, nhead=NHEAD, num_layers=NUM_LAYERS,
        hidden_dims=HIDDEN_DIMS, max_size=200, dropout=0.1).to(DEVICE)
    model, norm, mets = train_model(
        model, tM, tR, yt, vM, vR, yv, tsM, tsR, yts, model_type="rowcol_rank")

    # Save checkpoint
    ckpt_p = os.path.join(CKPT_DIR, "exp17_rowcol_rank.pt")
    ckpt = {
        "model_type": "rowcol_rank",
        "embed_dim": EMBED_DIM, "nhead": NHEAD, "num_layers": NUM_LAYERS,
        "config": {"hidden_dims": HIDDEN_DIMS},
        "max_size": 200,
        "max_rank": float(MAX_RANK),
        "model_state_dict": model.state_dict(),
        "normalizer": {"mean": norm.mu, "std": norm.s},
        "metrics": {k: float(v) for k, v in mets.items()},
        "train_size": NUM_TRAIN,
        "rollouts_per_target": ROLLOUTS,
        "label_type": "Minimum",
        "code": CODE_NAME,
    }
    torch.save(ckpt, ckpt_p)

    # MCTS evaluation
    print()
    mcts_r = eval_mcts("RowCol+Rank", model, norm, MAX_RANK, solver, init_m, nq, topo)

    # Summary
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Model: RowColTransformer + GF(2) rank")
    print(f"  R²={mets['R2']:.5f}  MAE={mets['MAE']:.2f}  r={mets['Pearson_r']:.4f}")
    cn = f"{mcts_r['cnot_mean']:.1f}±{mcts_r['cnot_std']:.1f}"
    dp = f"{mcts_r['depth_mean']:.1f}±{mcts_r['depth_std']:.1f}"
    print(f"  CNOT={cn}  Depth={dp}")
    print(f"  [VERIFIER] valid_rate={mcts_r['valid_rate']*100:.0f}%  "
          f"valid_cnot_mean={mcts_r['valid_cnot_mean']:.1f}  "
          f"invalid_count={mcts_r['invalid_count']}/"
          f"{sum(1 for _ in range(MCTS_RUNS))}  "
          f"z_syn={mcts_r['z_syndrome_mean']:.2f}  "
          f"x_syn={mcts_r['x_syndrome_mean']:.2f}  "
          f"logical={mcts_r['logical']}")
    print(f"  Checkpoint: {ckpt_p}")

    # CSV — extended to include verifier diagnostics alongside the
    # original metrics.  Original field names preserved for backward
    # compatibility with any downstream consumer.
    csv_p = os.path.join(RESULT_DIR, "exp17_summary.csv")
    with open(csv_p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model","cnot_mean","cnot_std",
                                           "depth_mean","depth_std","R2","MAE","Pearson_r",
                                           "valid_rate","valid_cnot_mean","valid_depth_mean",
                                           "invalid_count","logical",
                                           "z_syndrome_mean","x_syndrome_mean"])
        w.writeheader()
        w.writerow({
            "model": "RowCol+Rank",
            "cnot_mean": mcts_r["cnot_mean"], "cnot_std": mcts_r["cnot_std"],
            "depth_mean": mcts_r["depth_mean"], "depth_std": mcts_r["depth_std"],
            "R2": mets["R2"], "MAE": mets["MAE"], "Pearson_r": mets["Pearson_r"],
            "valid_rate": mcts_r["valid_rate"],
            "valid_cnot_mean": mcts_r["valid_cnot_mean"],
            "valid_depth_mean": mcts_r["valid_depth_mean"],
            "invalid_count": mcts_r["invalid_count"],
            "logical": mcts_r["logical"],
            "z_syndrome_mean": mcts_r["z_syndrome_mean"],
            "x_syndrome_mean": mcts_r["x_syndrome_mean"],
        })
    print(f"  Saved: {csv_p}")
    raw_csv = os.path.join(RESULT_DIR, "exp17_verifier_raw.csv")
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        fields = list(mcts_r["per_run"][0].keys()) if mcts_r["per_run"] else []
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(mcts_r["per_run"])
    print(f"  Verifier raw: {raw_csv}")


if __name__ == "__main__":
    main()
