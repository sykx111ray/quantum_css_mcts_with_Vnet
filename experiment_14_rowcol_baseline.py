"""
experiment_14_rowcol_baseline.py — Row/Column Transformer V-Net Baseline

Trains RowColValueNet (MatrixEncoder + MLP value head) on Surface d=5.
Compares with the flatten+MLP baseline from experiment_12.

Same pipeline as Exp12/13:
  - Shared dataset (same seed)
  - Minimum label (50-rollout)
  - MCTS evaluation

Key differences from Exp12/13:
  - Model: RowColValueNet (2-D matrix input) instead of SteaneValueNet (flat input)
  - Checkpoint: includes "model_type": "rowcol"
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
from value_network import RowColValueNet, matrix_to_input, compute_feature_dim

# ==============================================================================
# Configuration
# ==============================================================================
# CODE_NAME = "25_1_5_Rotated_Surface_Logical_0"
CODE_NAME = "81_1_9_Rotated_Surface_Logical_0"
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

# Matrix Encoder config
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


# ==============================================================================
# Helpers
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
    return sum(1 for g in c if g[0]=="CNOT")

def get_depth(c):
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
# Dataset generation (shared)
# ==============================================================================
def generate_shared_dataset(solver, init_m):
    N = NUM_TRAIN + NUM_VAL + NUM_TEST
    mats = []
    labs = np.zeros(N, dtype=np.float32)
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    print(f"Generating {N} states...")
    t0 = time.time()
    for i in range(N):
        s = sample_intermediate_state(init_m)
        mats.append(s.copy().astype(np.float32))
        labs[i] = compute_min_label(s, solver)
        if (i+1) % 500 == 0:
            print(f"  [{i+1:4d}/{N}]  min={labs[i]:.0f}  [{time.time()-t0:.0f}s]")
    print(f"  Done: {time.time()-t0:.0f}s")
    rng = np.random.RandomState(RNG_SEED+999)
    idx = rng.permutation(N)
    ms = [mats[i] for i in idx]
    ls = labs[idx]
    return (ms[:NUM_TRAIN], ms[NUM_TRAIN:NUM_TRAIN+NUM_VAL],
            ms[NUM_TRAIN+NUM_VAL:],
            ls[:NUM_TRAIN], ls[NUM_TRAIN:NUM_TRAIN+NUM_VAL],
            ls[NUM_TRAIN+NUM_VAL:])


# ==============================================================================
# Training
# ==============================================================================
def train_model(model, train_mat, y_train, val_mat, y_val, test_mat, y_test,
                model_type="rowcol", feature_names=None):
    norm = TargetNorm(); norm.fit(y_train)
    yt_n = norm.norm(y_train); yv_n = norm.norm(y_val)
    train_targets = yt_n.astype(np.float32)
    val_targets   = yv_n.astype(np.float32)

    if model_type == "rowcol":
        class Aligned2DDataset(Dataset):
            def __init__(self, matrices, targets):
                self.m = matrices; self.t = targets
            def __len__(self): return len(self.m)
            def __getitem__(self, i):
                return self.m[i], torch.tensor(self.t[i])

        train_ds2 = Aligned2DDataset([torch.from_numpy(m) for m in train_mat], train_targets)
        val_ds2   = Aligned2DDataset([torch.from_numpy(m) for m in val_mat], val_targets)
        test_mat_t = [torch.from_numpy(m) for m in test_mat]
    else:
        X_train = np.zeros((len(train_mat), compute_feature_dim(train_mat[0].shape, feature_names)), dtype=np.float32)
        X_val   = np.zeros((len(val_mat),   compute_feature_dim(val_mat[0].shape,   feature_names)), dtype=np.float32)
        X_test  = np.zeros((len(test_mat),  compute_feature_dim(test_mat[0].shape,  feature_names)), dtype=np.float32)
        for i, m in enumerate(train_mat): X_train[i] = matrix_to_input(m, feature_names=feature_names)
        for i, m in enumerate(val_mat):   X_val[i]   = matrix_to_input(m, feature_names=feature_names)
        for i, m in enumerate(test_mat):  X_test[i]  = matrix_to_input(m, feature_names=feature_names)

        train_ds2 = DatasetWrapper(X_train, train_targets)
        val_ds2   = DatasetWrapper(X_val, val_targets)
        test_mat_t = X_test

    train_ld = DataLoader(train_ds2, BATCH_SIZE, True,
                           collate_fn=collate_2d if model_type=="rowcol" else None)
    val_ld   = DataLoader(val_ds2, BATCH_SIZE, False,
                           collate_fn=collate_2d if model_type=="rowcol" else None)

    n_p = sum(p.numel() for p in model.parameters())
    print(f"  params={n_p:,}")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=PATIENCE//3)
    loss_fn = nn.MSELoss()

    best_v, best_ep, best_st, cnt = float("inf"), 0, None, 0
    for ep in range(1, EPOCHS+1):
        model.train(); tr_l = 0.0
        for batch in train_ld:
            if model_type == "rowcol":
                Xb, yb = batch[0].to(DEVICE), batch[1].to(DEVICE)
            else:
                Xb, yb = batch[0].to(DEVICE), batch[1].to(DEVICE)
            opt.zero_grad()
            l = loss_fn(model(Xb), yb); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tr_l += l.item()*len(yb)
        tr_l /= NUM_TRAIN

        model.eval(); vl_l = 0.0
        with torch.no_grad():
            for batch in val_ld:
                if model_type == "rowcol":
                    Xb, yb = batch[0].to(DEVICE), batch[1].to(DEVICE)
                else:
                    Xb, yb = batch[0].to(DEVICE), batch[1].to(DEVICE)
                vl_l += loss_fn(model(Xb), yb).item()*len(yb)
        vl_l /= NUM_VAL; sched.step(vl_l)

        if vl_l < best_v:
            best_v = vl_l; best_ep = ep; cnt = 0
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            cnt += 1
        if cnt >= PATIENCE: break

    model.load_state_dict(best_st)

    # Evaluate
    model.eval()
    if model_type == "rowcol":
        # Batch predict for 2-D
        all_preds = []
        bs = BATCH_SIZE
        for i in range(0, len(test_mat_t), bs):
            batch_m = torch.stack(test_mat_t[i:i+bs]).to(DEVICE)
            with torch.no_grad():
                all_preds.append(model(batch_m).cpu().numpy())
        yp_n = np.concatenate(all_preds)
    else:
        with torch.no_grad():
            X_t = torch.from_numpy(test_mat_t).to(DEVICE)
            yp_n = model(X_t).cpu().numpy()
    yp = norm.denorm(yp_n); yp = np.maximum(yp, 0.0)
    mets = compute_metrics(y_test, yp)
    print(f"  ep={best_ep:3d}  R²={mets['R2']:.5f}  MAE={mets['MAE']:.2f}  r={mets['Pearson_r']:.4f}")
    return model, norm, mets


class DatasetWrapper(Dataset):
    def __init__(self, X, y):
        self.X = X; self.y = y
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])


def collate_2d(batch):
    """Collate for 2-D matrix datasets (variable R, C currently not batched)."""
    matrices = [item[0] for item in batch]
    targets  = torch.tensor([item[1].item() if isinstance(item[1], torch.Tensor) else item[1] for item in batch])
    return torch.stack(matrices), targets


# ==============================================================================
# MCTS (simplified, self-contained)
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
    def __init__(s, env, slv, vn, norm, model_type, stats_recorder=None):
        s.e = env; s.s = slv; s.vn = vn; s.nm = norm; s.mt = model_type
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
        if s.mt == "rowcol":
            st = torch.from_numpy(n.m.astype(np.float32)).to(s.dev).unsqueeze(0)
        else:
            si = matrix_to_input(n.m, feature_names=["flatten", "rank"])
            st = torch.from_numpy(si).float().to(s.dev).unsqueeze(0)
        timer = s.stats.time_value_inference() if s.stats else None
        if timer: timer.__enter__()
        try:
            with torch.no_grad(): vn = s.vn(st).item()
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


def eval_mcts(ename, model, norm, model_type, solver, im, nq, topo):
    print(f"  MCTS: {ename}  ", end="", flush=True)
    model.eval()
    results = []
    # Pre-fetch CSS stabiliser / logical-Z strings for the diagnostic
    # verifier.  These are pure-Pauli strings (X- and Z-only) when
    # sourced from the registry; mixed-Pauli generators (e.g. the
    # [[5,1,3]] perfect code) are filtered out by `split_stabs`.
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
        log_csv = validation_log_path("exp14", rid)
        recorder = SearchStatsRecorder(log_csv, 100, _verify) if log_csv else None
        mcts = MCTS2D(
            env=Env2D(
                xs=[s for s in QuantumCodeRegistry.get_code(CODE_NAME)["stabs"]
                    if s.startswith("X")], nq=nq, topo=topo),
            slv=solver, vn=model, norm=norm, model_type=model_type,
            stats_recorder=recorder)
        t0 = time.perf_counter(); best = mcts.run(); rt = time.perf_counter()-t0
        ok = best is not None and len(best) > 0
        cn = get_cnot(best) if ok else 0; dp = get_depth(best) if ok else 0
        # --- diagnostic: stabilizer correctness verification (additive
        #     layer; does NOT change the cost / circuit being evaluated) ---
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
        row = {
            "cnot": cn, "depth": dp, "time": rt, "cost": mcts.bc, "ok": ok,
            "is_valid": is_valid,
            "syndrome_error": syn_err,
            "x_syndrome_error": v_diag.get("x_syndrome_error", float("inf")),
            "z_syndrome_error": v_diag.get("z_syndrome_error", float("inf")),
            "is_logical_zero": v_diag.get("is_logical_zero"),
            "v_num_cnot": v_diag.get("num_cnot", 0),
            "v_num_h": v_diag.get("num_h", 0),
        }
        if recorder:
            row.update(recorder.summary(count_tree_nodes(mcts.rt)))
        results.append(row)
        print(f"r{rid+1}={cn}{'V' if is_valid else 'I'}", end=" ", flush=True)
    cns = [r["cnot"] for r in results if r["ok"]]
    dns = [r["depth"] for r in results if r["ok"]]
    c_m = np.mean(cns) if cns else 0; c_s = np.std(cns, ddof=1) if len(cns)>1 else 0
    d_m = np.mean(dns) if dns else 0; d_s = np.std(dns, ddof=1) if len(dns)>1 else 0
    # Verifier diagnostic aggregates (computed over the runs that produced
    # a non-empty circuit; failures of MCTS itself are counted separately).
    valid_flags = [r["is_valid"] for r in results if r["ok"]]
    valid_cns = [r["cnot"] for r in results if r["ok"] and r["is_valid"]]
    valid_dns = [r["depth"] for r in results if r["ok"] and r["is_valid"]]
    valid_rate = float(np.mean(valid_flags)) if valid_flags else 0.0
    valid_cnot_mean = float(np.mean(valid_cns)) if valid_cns else 0.0
    valid_depth_mean = float(np.mean(valid_dns)) if valid_dns else 0.0
    invalid_count = int(sum(1 for r in results if r["ok"] and not r["is_valid"]))
    z_err_mean = float(np.mean([r["z_syndrome_error"] for r in results
                                if r["ok"] and r["z_syndrome_error"] != float("inf")])) \
        if any(r["ok"] and r["z_syndrome_error"] != float("inf") for r in results) else 0.0
    x_err_mean = float(np.mean([r["x_syndrome_error"] for r in results
                                if r["ok"] and r["x_syndrome_error"] != float("inf")])) \
        if any(r["ok"] and r["x_syndrome_error"] != float("inf") for r in results) else 0.0
    print(f"→ {c_m:.1f}±{c_s:.1f}  [valid={valid_rate*100:.0f}%]")
    verifier_summary = aggregate_verifier_results(results)
    return dict(ename=ename, cnot_mean=c_m, cnot_std=c_s, depth_mean=d_m, depth_std=d_s,
                valid_rate=valid_rate, valid_cnot_mean=valid_cnot_mean,
                valid_depth_mean=valid_depth_mean,
                invalid_count=invalid_count, z_syndrome_mean=z_err_mean,
                x_syndrome_mean=x_err_mean,
                logical=verifier_summary["logical"],
                verifier_summary=verifier_summary,
                per_run=results)


# ==============================================================================
# Main
# ==============================================================================
def main():
    sys.stdout.reconfigure(line_buffering=True)
    print("="*60)
    print("Experiment 14: RowCol Transformer Baseline")
    print("="*60)
    print(f"Code: {CODE_NAME}, Label: Minimum ({ROLLOUTS}-rollout)")

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
    im = np.zeros((len(xs), nq), dtype=int)
    for i, s in enumerate(xs):
        for p in s.split("*"): im[i, int(p.strip()[1:])] = 1
    topo = get_all_to_all_edges(nq)
    solver = HeuristicRolloutSolver(topo, nq, code_name=CODE_NAME)

    # Dataset
    tM, vM, tsM, yt, yv, yts = generate_shared_dataset(solver, im)
    print(f"  y_train: mean={yt.mean():.1f} std={yt.std():.1f}")

    # Train RowCol model
    print(f"\n{'='*60}")
    print("Training: RowColTransformer")
    print(f"{'='*60}")

    model = RowColValueNet(
        embed_dim=EMBED_DIM, nhead=NHEAD, num_layers=NUM_LAYERS,
        hidden_dims=HIDDEN_DIMS, max_size=200, dropout=0.1).to(DEVICE)
    model, norm, mets = train_model(model, tM, yt, vM, yv, tsM, yts,
                                     model_type="rowcol")

    # Save checkpoint (compatible with quantum_mcts.py _load_value_network)
    ckpt = {
        "model_type": "rowcol",
        "embed_dim": EMBED_DIM, "nhead": NHEAD, "num_layers": NUM_LAYERS,
        "config": {"hidden_dims": HIDDEN_DIMS},
        "max_size": 200,
        "model_state_dict": model.state_dict(),
        "normalizer": {"mean": norm.mu, "std": norm.s},
        "metrics": {k: float(v) for k, v in mets.items()},
        "train_size": NUM_TRAIN,
        "rollouts_per_target": ROLLOUTS,
        "label_type": "Minimum",
        "code": CODE_NAME,
    }
    ckpt_p = os.path.join(CKPT_DIR, "exp14_rowcol.pt")
    torch.save(ckpt, ckpt_p)

    # MCTS evaluation
    mcts_r = eval_mcts("RowCol", model, norm, "rowcol", solver, im, nq, topo)

    # Summary
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    cn = f"{mcts_r['cnot_mean']:.1f}±{mcts_r['cnot_std']:.1f}"
    dp = f"{mcts_r['depth_mean']:.1f}±{mcts_r['depth_std']:.1f}"
    print(f"  Model: RowColTransformer")
    print(f"  R²={mets['R2']:.5f}  MAE={mets['MAE']:.2f}  r={mets['Pearson_r']:.4f}")
    print(f"  CNOT={cn}  Depth={dp}")
    print(f"  [VERIFIER] valid_rate={mcts_r['valid_rate']*100:.0f}%  "
          f"valid_cnot_mean={mcts_r['valid_cnot_mean']:.1f}  "
          f"invalid_count={mcts_r['invalid_count']}/"
          f"{sum(1 for _ in range(MCTS_RUNS))}  "
          f"z_syn={mcts_r['z_syndrome_mean']:.2f}  "
          f"x_syn={mcts_r['x_syndrome_mean']:.2f}  "
          f"logical={mcts_r['logical']}")
    print(f"  Checkpoint: {ckpt_p}")

    diag_csv = os.path.join(RESULT_DIR, "exp14_rowcol_verifier_summary.csv")
    with open(diag_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "cnot_mean", "cnot_std", "depth_mean", "depth_std",
            "valid_rate", "valid_cnot_mean", "valid_depth_mean",
            "invalid_count", "logical",
            "z_syndrome_mean", "x_syndrome_mean", "R2", "MAE", "Pearson_r"])
        w.writeheader()
        w.writerow({
            "model": "RowCol",
            "cnot_mean": mcts_r["cnot_mean"], "cnot_std": mcts_r["cnot_std"],
            "depth_mean": mcts_r["depth_mean"], "depth_std": mcts_r["depth_std"],
            "valid_rate": mcts_r["valid_rate"],
            "valid_cnot_mean": mcts_r["valid_cnot_mean"],
            "valid_depth_mean": mcts_r["valid_depth_mean"],
            "invalid_count": mcts_r["invalid_count"],
            "logical": mcts_r["logical"],
            "z_syndrome_mean": mcts_r["z_syndrome_mean"],
            "x_syndrome_mean": mcts_r["x_syndrome_mean"],
            "R2": mets["R2"], "MAE": mets["MAE"], "Pearson_r": mets["Pearson_r"],
        })
    raw_diag_csv = os.path.join(RESULT_DIR, "exp14_rowcol_verifier_raw.csv")
    with open(raw_diag_csv, "w", newline="", encoding="utf-8") as f:
        fields = list(mcts_r["per_run"][0].keys()) if mcts_r["per_run"] else []
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(mcts_r["per_run"])
    print(f"  Verifier summary: {diag_csv}")
    print(f"  Verifier raw: {raw_diag_csv}")


if __name__ == "__main__":
    main()
