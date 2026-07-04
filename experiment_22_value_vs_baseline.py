"""
Experiment 22: Value Network vs Baseline MCTS

Question:
    Does Value Network improve MCTS?

This is a controlled evaluation script only. It does not train any model,
does not modify the MCTS core algorithm, and does not change verifier logic.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from collections import Counter

import numpy as np
import torch

from quantum_registry import QuantumCodeRegistry
from quantum_synthesizer import HeuristicRolloutSolver, build_css_logical_zero_prep
from utils.circuit_verifier import split_stabs, verify_css_circuit
from value_network import (
    CostNormalizer,
    RowColRankValueNet,
    RowColValueNet,
    SteaneValueNet,
    _gf2_rank,
    matrix_to_input,
)


CODE_NAME = "81_1_9_Rotated_Surface_Logical_0"

N_RUNS = 20
SEED_BASE = 1000
MCTS_ITERATIONS = 2000
MCTS_ACTIONS = 10
ROLLOUTS_PER_TARGET = 50
VERIFY_INTERVAL = 100
RECORD_INTERVAL = 100
UCT_C = 1.5
DEFAULT_VALUE_CKPT = os.path.join("checkpoints", "exp17_rowcol_rank.pt")
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TRACE_OUTPUT = os.path.join(RESULT_DIR, "exp22_trace.csv")
TRACE_SUMMARY = os.path.join(RESULT_DIR, "trace_summary.txt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _disable_flash_sdp() -> None:
    """Avoid flash-attention kernel probes that spam FATAL on mismatched GPU/PyTorch builds."""
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)


def _resolve_device(preferred: Optional[str] = None) -> torch.device:
    if preferred == "cpu":
        return torch.device("cpu")

    cuda_available = torch.cuda.is_available()

    if preferred == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDA requested via --device cuda but torch.cuda.is_available() is False")
        _disable_flash_sdp()
        return torch.device("cuda")

    if not cuda_available:
        return torch.device("cpu")

    _disable_flash_sdp()
    try:
        torch.zeros(1, device="cuda")
        return torch.device("cuda")
    except Exception:
        return torch.device("cpu")


def get_all_to_all_edges(n: int):
    return list(itertools.combinations(range(n), 2))


def get_cnot_count(circuit: Optional[Iterable]) -> int:
    if not circuit:
        return 0
    return sum(1 for g in circuit if g and g[0] == "CNOT")


def get_depth(circuit: Optional[Iterable]) -> int:
    return len(list(circuit)) if circuit else 0


def _mean_std(vals: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def _t_ppf_975(df: float) -> float:
    """Approximate t_0.975(df) with numpy only (no scipy)."""
    if df <= 2:
        return 12.7 if df == 1 else 4.303
    if df >= 30:
        return 1.96
    return 1.96 + 2.0 / df  # conservative for small df


def _ci95(vals: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, mean
    sem = float(arr.std(ddof=1) / math.sqrt(len(arr)))
    half = float(_t_ppf_975(len(arr) - 1) * sem)
    return mean - half, mean + half


def _cohen_dz(diffs: Sequence[float]) -> float:
    arr = np.asarray(diffs, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    sd = float(arr.std(ddof=1))
    if sd == 0.0:
        return 0.0
    return float(arr.mean() / sd)


def _safe_wilcoxon(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Wilcoxon signed-rank test (numpy only, no scipy)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    d = d[np.isfinite(d)]
    d = d[d != 0.0]
    n = len(d)
    if n == 0:
        return float("nan"), float("nan")
    ranks = np.argsort(np.abs(d))
    signed_ranks = np.sign(d[ranks]) * (np.arange(n) + 1)
    T = signed_ranks[signed_ranks > 0].sum()
    z = (T - n * (n + 1) / 4) / math.sqrt(n * (n + 1) * (2 * n + 1) / 24) if n > 1 else 0.0
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return float(T), float(p)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF approximation."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _safe_ttest(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Paired t-test (numpy only, no scipy)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return float("nan"), float("nan")
    t = d.mean() / (d.std(ddof=1) / math.sqrt(n))
    df = n - 1
    p = 2.0 * (1.0 - _norm_cdf(abs(t))) if df > 30 else 2.0 * _tcdf_surv(abs(t), df)
    return float(t), float(p)


def _tcdf_surv(t: float, df: float) -> float:
    """Conservative approx of Student t survival for small df."""
    return _norm_cdf(-t)  # approximation for df < 30; conservative


def _write_csv(path: str, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _normalise_bool(v) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, str):
        if v.lower() in {"true", "1", "yes"}:
            return True
        if v.lower() in {"false", "0", "no"}:
            return False
        return None
    return bool(v)


def _extract_num_qubits(cfg: Dict) -> int:
    max_idx = -1
    for stab in cfg["stabs"]:
        for p in stab.split("*"):
            token = p.strip()
            if not token:
                continue
            max_idx = max(max_idx, int(token[1:]))
    return max_idx + 1


class SurfaceEnv:
    def __init__(self, x_stabs: Sequence[str], num_qubits: int, topo_edges):
        import networkx as nx

        self.num_qubits = num_qubits
        self.graph = nx.Graph(topo_edges)
        self.M = len(x_stabs)
        self.init_matrix = np.zeros((self.M, num_qubits), dtype=int)
        for i, stab in enumerate(x_stabs):
            for p in stab.split("*"):
                self.init_matrix[i, int(p.strip()[1:])] = 1

    def apply_cnot(self, matrix, ctrl, targ):
        new_mat = matrix.copy()
        new_mat[:, targ] ^= new_mat[:, ctrl]
        return new_mat

    def get_actions(self, matrix, prefix):
        actions = []
        active = np.where(matrix.any(axis=0))[0]
        layer = len(prefix)
        group = layer % 4
        for q in active:
            if q % 4 != group:
                continue
            for nb in self.graph.neighbors(q):
                if nb >= self.num_qubits or q >= self.num_qubits:
                    continue
                actions.extend([("CNOT", q, nb), ("CNOT", nb, q)])
        actions.append(("ID", -1, -1))
        if prefix:
            actions = [a for a in actions if a != prefix[-1]]
        actions = list(set(actions))
        random.shuffle(actions)
        return actions[:MCTS_ACTIONS]


class Node:
    __slots__ = ("prefix", "matrix", "parent", "children", "untried",
                 "fully_expanded", "visits", "total_cost", "action", "node_id")

    def __init__(self, prefix, matrix, parent=None, action=None, node_id=0):
        self.prefix = prefix
        self.matrix = matrix
        self.parent = parent
        self.children = []
        self.untried = []
        self.fully_expanded = False
        self.visits = 0
        self.total_cost = 0.0
        self.action = action
        self.node_id = node_id


@dataclass
class ArmRun:
    seed: int
    arm: str
    cnot: int
    depth: int
    runtime_s: float
    is_valid: bool
    logical_ok: Optional[bool]
    x_syndrome: float
    z_syndrome: float
    syndrome_error: float
    first_valid_iteration: Optional[int]
    first_best_iteration: Optional[int]
    search_time: float
    value_inference_time: float
    simulation_time: float
    node_count: int
    best_iteration: Optional[int]


class ConvergenceRecorder:
    def __init__(self, record_every: int = RECORD_INTERVAL):
        self.record_every = record_every
        self.rows: List[Dict] = []
        self.first_valid_iteration: Optional[int] = None
        self.first_best_iteration: Optional[int] = None
        self._best_signature: Optional[Tuple[int, int]] = None

    def note_best(self, iteration: int, circuit: Optional[Iterable]) -> None:
        if not circuit:
            return
        sig = (get_cnot_count(circuit), get_depth(circuit))
        if sig != self._best_signature:
            self._best_signature = sig
            self.first_best_iteration = int(iteration)

    def note_valid(self, iteration: int) -> None:
        if self.first_valid_iteration is None:
            self.first_valid_iteration = int(iteration)

    def record(self, iteration: int, circuit: Optional[Iterable], verdict: Dict,
               runtime_s: float, arm: str, seed: int) -> None:
        if iteration % self.record_every != 0:
            return
        self.rows.append({
            "seed": seed,
            "arm": arm,
            "iteration": iteration,
            "best_cnot": get_cnot_count(circuit),
            "best_depth": get_depth(circuit),
            "is_valid": verdict.get("is_valid"),
            "logical_ok": verdict.get("logical_ok"),
            "x_syndrome": verdict.get("x_syndrome"),
            "z_syndrome": verdict.get("z_syndrome"),
            "runtime_s": runtime_s,
            "first_valid_iteration": self.first_valid_iteration,
            "first_best_iteration": self.first_best_iteration,
        })


class TraceRecorder:
    def __init__(self, output_path: Optional[str] = None):
        self.output_path = output_path
        self.rows: List[Dict] = []
        self.action_counts: Counter = Counter()

    @staticmethod
    def _fmt_action(action) -> str:
        if action is None:
            return ""
        if isinstance(action, (tuple, list)) and len(action) >= 3:
            return f"{action[0]}:{action[1]}:{action[2]}"
        return str(action)

    def record_iteration(self, run_id: int, step: int, node_id: Optional[int], parent_id: Optional[int],
                         mode: str, selected_action: Optional[Iterable], visit_entropy: float,
                         uct_scores_mean: float, uct_scores_std: float,
                         rollout_mean: Optional[float], rollout_std: Optional[float],
                         rollout_min: Optional[float], value_pred: Optional[float],
                         value_zscore: Optional[float]) -> None:
        action_key = self._fmt_action(selected_action)
        if action_key:
            self.action_counts[action_key] += 1
        top1 = max(self.action_counts.values(), default=0)
        top3 = sum(sorted(self.action_counts.values(), reverse=True)[:3]) if self.action_counts else 0
        self.rows.append({
            "run_id": run_id,
            "step": step,
            "node_id": node_id,
            "parent_id": parent_id,
            "mode": mode,
            "selected_action": action_key,
            "visit_entropy": float(visit_entropy or 0.0),
            "uct_scores_mean": float(uct_scores_mean or 0.0),
            "uct_scores_std": float(uct_scores_std or 0.0),
            "rollout_mean": "" if rollout_mean is None else float(rollout_mean),
            "rollout_std": "" if rollout_std is None else float(rollout_std),
            "rollout_min": "" if rollout_min is None else float(rollout_min),
            "value_pred": "" if value_pred is None else float(value_pred),
            "value_zscore": "" if value_zscore is None else float(value_zscore),
            "action_counts_top1": int(top1),
            "action_counts_top3": int(top3),
        })

    def write_csv(self, path: Optional[str] = None) -> None:
        out_path = path or self.output_path
        if not out_path:
            return
        fieldnames = [
            "run_id", "step", "node_id", "parent_id", "mode", "selected_action",
            "visit_entropy", "uct_scores_mean", "uct_scores_std",
            "rollout_mean", "rollout_std", "rollout_min", "value_pred", "value_zscore",
            "action_counts_top1", "action_counts_top3"
        ]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

    def write_summary(self, path: Optional[str] = None) -> None:
        out_path = path or self.output_path.replace(".csv", "_summary.txt")
        if not out_path:
            return
        rows = self.rows
        if not rows:
            text = "No trace rows collected."
        else:
            baseline = [r for r in rows if r["mode"] == "baseline"]
            value = [r for r in rows if r["mode"] == "value"]
            def _mean(vals):
                vals = [float(v) for v in vals if v not in (None, "")]
                return float(np.mean(vals)) if vals else float("nan")
            text = "\n".join([
                "Experiment 22 Value Influence Trace Summary",
                f"rows={len(rows)}",
                f"baseline_rows={len(baseline)}",
                f"value_rows={len(value)}",
                f"baseline_entropy_mean={_mean([r['visit_entropy'] for r in baseline]):.4f}",
                f"value_entropy_mean={_mean([r['visit_entropy'] for r in value]):.4f}",
                f"baseline_action_top1_mean={_mean([r['action_counts_top1'] for r in baseline]):.4f}",
                f"value_action_top1_mean={_mean([r['action_counts_top1'] for r in value]):.4f}",
                f"baseline_rollout_mean={_mean([r['rollout_mean'] for r in baseline]):.4f}",
                f"value_pred_mean={_mean([r['value_pred'] for r in value]):.4f}",
            ])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")


class Exp22MCTS:
    def __init__(self, env, solver, use_value=False, value_net=None,
                 normalizer=None, model_type=None, max_rank=1.0,
                 rollouts=ROLLOUTS_PER_TARGET, verifier=None,
                 seed=0, arm_name="Baseline", iterations=MCTS_ITERATIONS,
                 record_interval=RECORD_INTERVAL, verify_interval=VERIFY_INTERVAL,
                 trace_recorder=None, run_id=0):
        self.env = env
        self.solver = solver
        self.use_value = use_value
        self.value_net = value_net
        self.normalizer = normalizer
        self.model_type = model_type
        self.max_rank = float(max_rank)
        self.rollouts = int(rollouts)
        self.verifier = verifier
        self.seed = seed
        self.arm_name = arm_name
        self.iterations = int(iterations)
        self.record_interval = int(record_interval)
        self.verify_interval = int(verify_interval)
        self.device = DEVICE
        self.trace_recorder = trace_recorder
        self.run_id = int(run_id)
        self._last_selection_info: Optional[Dict] = None
        self._last_eval_info: Optional[Dict] = None
        self._next_node_id = 1

        self.root = Node([], self.env.init_matrix, node_id=0)
        self.root.untried = self.env.get_actions(self.root.matrix, [])

        self.best_circuit = None
        self.best_cost = float("inf")
        self.best_iteration = None
        self._eval_count = 0
        self.convergence = ConvergenceRecorder(self.record_interval)
        self.search_time = 0.0
        self.value_inference_time = 0.0
        self.simulation_time = 0.0

    def _select(self, node):
        self._last_selection_info = {
            "selected_action": None,
            "visit_entropy": 0.0,
            "uct_scores_mean": 0.0,
            "uct_scores_std": 0.0,
            "node_id": node.node_id,
            "parent_id": node.parent.node_id if node.parent is not None else None,
        }
        while node.children and node.fully_expanded:
            uct_scores = []
            for c in node.children:
                score = (c.total_cost / max(c.visits, 1e-6)) - UCT_C * math.sqrt(
                    math.log(max(node.visits, 1)) / max(c.visits, 1e-6)
                )
                uct_scores.append(score)
            child = min(node.children, key=lambda c: (c.total_cost / max(c.visits, 1e-6))
                        - UCT_C * math.sqrt(math.log(max(node.visits, 1)) / max(c.visits, 1e-6)))
            visits = [max(c.visits, 0) for c in node.children]
            total = sum(visits)
            if total > 0:
                probs = [v / total for v in visits]
                entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs if p > 0)
            else:
                entropy = 0.0
            self._last_selection_info = {
                "selected_action": child.action,
                "visit_entropy": float(entropy),
                "uct_scores_mean": float(np.mean(uct_scores)) if uct_scores else 0.0,
                "uct_scores_std": float(np.std(uct_scores)) if len(uct_scores) > 1 else 0.0,
                "node_id": node.node_id,
                "parent_id": node.parent.node_id if node.parent is not None else None,
            }
            node = child
        return node

    def _expand(self, node):
        if not node.untried:
            return node
        action = node.untried.pop()
        if not node.untried:
            node.fully_expanded = True
        child = Node(
            node.prefix + [action],
            self.env.apply_cnot(node.matrix, action[1], action[2]) if action[0] == "CNOT"
            else node.matrix.copy(),
            parent=node,
            action=action,
            node_id=self._next_node_id,
        )
        self._next_node_id += 1
        child.untried = self.env.get_actions(child.matrix, child.prefix)
        node.children.append(child)
        return child

    def _rollout_costs(self, node) -> Tuple[List[float], float, Optional[List]]:
        rollout_values: List[float] = []
        best_cost = float("inf")
        best_circuit = None
        for _ in range(self.rollouts):
            gates, pivots = self.solver.solve_remainder(node.matrix, randomize=True)
            if gates is None or pivots is None:
                continue
            circuit = build_css_logical_zero_prep(node.prefix + gates, pivots)
            cost = len(circuit)
            rollout_values.append(cost)
            if cost < best_cost:
                best_cost = cost
                best_circuit = circuit
        return rollout_values, best_cost, best_circuit

    def _exact_rollout_cost(self, node) -> Tuple[float, Optional[List]]:
        _, best_cost, best_circuit = self._rollout_costs(node)
        return best_cost, best_circuit

    def _estimate_cost(self, node) -> float:
        if self.use_value:
            rollout_values, _, _ = self._rollout_costs(node)
            if self.model_type == "rowcol_rank":
                matrix = torch.from_numpy(node.matrix.astype(np.float32)).float().to(self.device).unsqueeze(0)
                rank = torch.tensor([[_gf2_rank(node.matrix) / max(self.max_rank, 1.0)]],
                                    dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    pred = self.value_net(matrix, rank).item()
            elif self.model_type == "rowcol":
                matrix = torch.from_numpy(node.matrix.astype(np.float32)).float().to(self.device).unsqueeze(0)
                with torch.no_grad():
                    pred = self.value_net(matrix).item()
            else:
                inp = matrix_to_input(node.matrix, include_features=True)
                matrix = torch.from_numpy(inp).float().to(self.device).unsqueeze(0)
                with torch.no_grad():
                    pred = self.value_net(matrix).item()
            remaining = max(0.0, self.normalizer.denormalize(pred))
            self._last_eval_info = {
                "rollout_values": rollout_values,
                "value_pred": float(remaining - len(node.prefix)),
            }
            return len(node.prefix) + remaining

        rollout_values, best_cost, _ = self._rollout_costs(node)
        self._last_eval_info = {
            "rollout_values": rollout_values,
            "value_pred": None,
        }
        return best_cost

    def _maybe_update_best(self, iteration: int, node) -> None:
        with torch.no_grad():
            cost, circuit = self._exact_rollout_cost(node)
        if circuit is not None and cost < self.best_cost:
            self.best_cost = cost
            self.best_circuit = circuit
            self.best_iteration = int(iteration)
            self.convergence.note_best(iteration, circuit)

    def _backprop(self, node, cost):
        while node is not None:
            node.visits += 1
            node.total_cost += cost
            node = node.parent

    def run(self):
        t0 = time.perf_counter()
        progress_interval = max(1, self.iterations // 10)
        print(f"  [{self.arm_name} seed={self.seed}] ", end="", flush=True)
        for it in range(1, self.iterations + 1):
            leaf = self._select(self.root)
            child = self._expand(leaf)
            eval_t0 = time.perf_counter()
            cost = self._estimate_cost(child)
            self.value_inference_time += time.perf_counter() - eval_t0 if self.use_value else 0.0
            if not self.use_value:
                self.simulation_time += time.perf_counter() - eval_t0
            if self.trace_recorder is not None and self._last_selection_info is not None:
                rollout_values = self._last_eval_info.get("rollout_values") if self._last_eval_info else None
                value_pred = self._last_eval_info.get("value_pred") if self._last_eval_info else None
                rollout_values = rollout_values or []
                if rollout_values:
                    rollout_mean = float(np.mean(rollout_values))
                    rollout_std = float(np.std(rollout_values)) if len(rollout_values) > 1 else 0.0
                    rollout_min = float(np.min(rollout_values))
                else:
                    rollout_mean = rollout_std = rollout_min = None
                if rollout_values and value_pred is not None:
                    if rollout_std is None or rollout_std == 0.0:
                        value_zscore = 0.0
                    else:
                        value_zscore = float((value_pred - rollout_mean) / rollout_std)
                else:
                    value_zscore = None
                self.trace_recorder.record_iteration(
                    run_id=self.run_id,
                    step=it,
                    node_id=child.node_id,
                    parent_id=child.parent.node_id if child.parent is not None else None,
                    mode="value" if self.use_value else "baseline",
                    selected_action=self._last_selection_info.get("selected_action"),
                    visit_entropy=self._last_selection_info.get("visit_entropy", 0.0),
                    uct_scores_mean=self._last_selection_info.get("uct_scores_mean", 0.0),
                    uct_scores_std=self._last_selection_info.get("uct_scores_std", 0.0),
                    rollout_mean=rollout_mean,
                    rollout_std=rollout_std,
                    rollout_min=rollout_min,
                    value_pred=value_pred,
                    value_zscore=value_zscore,
                )
            self._eval_count += 1
            self._backprop(child, cost)

            if it % progress_interval == 0:
                print(".", end="", flush=True)

            if self._eval_count % self.verify_interval == 0:
                self._maybe_update_best(it, child)
                verdict = self._verify_current_best()
                if verdict.get("is_valid") and self.convergence.first_valid_iteration is None:
                    self.convergence.note_valid(it)
                self.convergence.record(
                    it, self.best_circuit, verdict, time.perf_counter() - t0,
                    self.arm_name, self.seed)

        if self.best_circuit is None:
            self._extract_best()

        self.search_time = time.perf_counter() - t0
        if self.best_circuit is not None and self.convergence.first_best_iteration is None:
            self.convergence.note_best(self.iterations, self.best_circuit)
        final_verdict = self._verify_current_best()
        if final_verdict.get("is_valid") and self.convergence.first_valid_iteration is None:
            self.convergence.note_valid(self.iterations)
        cnot = get_cnot_count(self.best_circuit)
        print(f" CNOT={cnot} ({self.search_time:.1f}s)", flush=True)
        return self.best_circuit, final_verdict

    def _extract_best(self):
        best_node = None
        best_avg = float("inf")
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.visits > 0:
                avg = node.total_cost / node.visits
                if avg < best_avg:
                    best_avg = avg
                    best_node = node
            stack.extend(node.children)
        if best_node is None:
            return None
        _, circuit = self._exact_rollout_cost(best_node)
        self.best_circuit = circuit
        self.best_cost = len(circuit) if circuit is not None else float("inf")
        self.best_iteration = self.iterations
        return circuit

    def _verify_current_best(self) -> Dict:
        if self.best_circuit is None:
            return {
                "is_valid": False,
                "logical_ok": None,
                "x_syndrome": float("inf"),
                "z_syndrome": float("inf"),
                "syndrome_error": float("inf"),
            }
        is_valid, syn_err, diag = self.verifier(self.best_circuit)
        return {
            "is_valid": bool(is_valid),
            "logical_ok": _normalise_bool(diag.get("is_logical_zero")),
            "x_syndrome": float(diag.get("x_syndrome_error", float("inf"))),
            "z_syndrome": float(diag.get("z_syndrome_error", float("inf"))),
            "syndrome_error": float(syn_err),
        }


def _load_value_network(ckpt_path: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Value checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model_type = ckpt.get("model_type", None)
    # Auto-detect model_type from state_dict keys when missing (e.g. Exp20 L0)
    if model_type is None:
        sd_keys = list(ckpt["model_state_dict"].keys())
        if any("net." in k for k in sd_keys):
            model_type = "steane"
        elif any("matrix_encoder" in k for k in sd_keys):
            if any("rank" in k for k in sd_keys):
                model_type = "rowcol_rank"
            else:
                model_type = "rowcol"
        else:
            model_type = "steane"
    if model_type == "rowcol_rank":
        model = RowColRankValueNet(
            embed_dim=ckpt.get("embed_dim", 128),
            nhead=ckpt.get("nhead", 4),
            num_layers=ckpt.get("num_layers", 2),
            hidden_dims=ckpt["config"]["hidden_dims"],
            max_size=ckpt.get("max_size", 200),
            dropout=ckpt.get("dropout", 0.1),
        )
    elif model_type == "rowcol":
        model = RowColValueNet(
            embed_dim=ckpt.get("embed_dim", 128),
            nhead=ckpt.get("nhead", 4),
            num_layers=ckpt.get("num_layers", 2),
            hidden_dims=ckpt["config"]["hidden_dims"],
            max_size=ckpt.get("max_size", 200),
            dropout=ckpt.get("dropout", 0.1),
        )
    else:
        model = SteaneValueNet(
            input_dim=ckpt["input_dim"],
            hidden_dims=ckpt["config"]["hidden_dims"],
            dropout=ckpt.get("dropout", 0.0),
        )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    normalizer = CostNormalizer()
    normalizer.load_state_dict(ckpt["normalizer"])
    max_rank = float(ckpt.get("max_rank", 1.0))
    return model, normalizer, model_type, max_rank


def _verifier_for(config):
    stabs_X, stabs_Z = split_stabs(config["stabs"])
    logicals = config.get("logicals", [])

    def _verify(circuit):
        is_valid, syn_err, diag = verify_css_circuit(circuit, stabs_X, stabs_Z, logicals)
        return is_valid, syn_err, {
            "x_syndrome_error": diag.get("x_syndrome_error", float("inf")),
            "z_syndrome_error": diag.get("z_syndrome_error", float("inf")),
            "is_logical_zero": diag.get("is_logical_zero"),
        }

    return _verify


def _summarise_arm(rows: Sequence[ArmRun]) -> Dict[str, float]:
    cnot = [r.cnot for r in rows]
    depth = [r.depth for r in rows]
    runtime = [r.runtime_s for r in rows]
    valid = [r.is_valid for r in rows]
    return {
        "cnot_mean": _mean_std(cnot)[0],
        "cnot_std": _mean_std(cnot)[1],
        "depth_mean": _mean_std(depth)[0],
        "depth_std": _mean_std(depth)[1],
        "runtime_mean": _mean_std(runtime)[0],
        "runtime_std": _mean_std(runtime)[1],
        "valid_rate": float(np.mean(valid)) if valid else 0.0,
    }


def _paired_stats(base: Sequence[float], value: Sequence[float]) -> Dict[str, float]:
    base = np.asarray(base, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    diffs = value - base
    t_stat, t_p = _safe_ttest(value, base)
    w_stat, w_p = _safe_wilcoxon(value, base)
    ci_low, ci_high = _ci95(diffs)
    return {
        "delta_mean": float(diffs.mean()) if len(diffs) else 0.0,
        "delta_std": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "paired_t": t_stat,
        "paired_t_p": t_p,
        "wilcoxon": w_stat,
        "wilcoxon_p": w_p,
        "cohen_dz": _cohen_dz(diffs),
    }


def run_arm(arm: str, seed: int, env, solver, verifier, iterations: int,
            rollouts: int, record_interval: int, verify_interval: int,
            value_cfg=None, trace_recorder=None, run_id=0) -> Tuple[ArmRun, List[Dict]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if value_cfg is not None:
        model, normalizer, model_type, max_rank = value_cfg
        use_value = True
    else:
        model = normalizer = model_type = max_rank = None
        use_value = False

    mcts = Exp22MCTS(
        env=env,
        solver=solver,
        use_value=use_value,
        value_net=model,
        normalizer=normalizer,
        model_type=model_type,
        max_rank=max_rank if max_rank is not None else 1.0,
        rollouts=rollouts,
        verifier=verifier,
        seed=seed,
        arm_name=arm,
        iterations=iterations,
        record_interval=record_interval,
        verify_interval=verify_interval,
        trace_recorder=trace_recorder,
        run_id=run_id,
    )

    t0 = time.perf_counter()
    best_circuit, final_verdict = mcts.run()
    runtime = time.perf_counter() - t0
    cnot = get_cnot_count(best_circuit)
    depth = get_depth(best_circuit)
    row = ArmRun(
        seed=seed,
        arm=arm,
        cnot=cnot,
        depth=depth,
        runtime_s=runtime,
        is_valid=bool(final_verdict["is_valid"]),
        logical_ok=final_verdict["logical_ok"],
        x_syndrome=final_verdict["x_syndrome"],
        z_syndrome=final_verdict["z_syndrome"],
        syndrome_error=final_verdict["syndrome_error"],
        first_valid_iteration=mcts.convergence.first_valid_iteration,
        first_best_iteration=mcts.convergence.first_best_iteration,
        search_time=mcts.search_time,
        value_inference_time=mcts.value_inference_time,
        simulation_time=mcts.simulation_time,
        node_count=sum(1 for _ in _walk_tree(mcts.root)),
        best_iteration=mcts.best_iteration,
    )

    convergence_rows = []
    for r in mcts.convergence.rows:
        row_copy = dict(r)
        row_copy["seed"] = seed
        row_copy["arm"] = arm
        convergence_rows.append(row_copy)
    if not convergence_rows:
        convergence_rows.append({
            "seed": seed,
            "arm": arm,
            "iteration": MCTS_ITERATIONS,
            "best_cnot": cnot,
            "best_depth": depth,
            "is_valid": row.is_valid,
            "logical_ok": row.logical_ok,
            "x_syndrome": row.x_syndrome,
            "z_syndrome": row.z_syndrome,
            "runtime_s": runtime,
            "first_valid_iteration": row.first_valid_iteration,
            "first_best_iteration": row.first_best_iteration,
        })
    return row, convergence_rows


def _walk_tree(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--value_ckpt", default=DEFAULT_VALUE_CKPT)
    parser.add_argument("--runs", type=int, default=N_RUNS)
    parser.add_argument("--iterations", type=int, default=MCTS_ITERATIONS)
    parser.add_argument("--rollouts", type=int, default=ROLLOUTS_PER_TARGET)
    parser.add_argument("--record_interval", type=int, default=RECORD_INTERVAL)
    parser.add_argument("--verify_interval", type=int, default=VERIFY_INTERVAL)
    parser.add_argument("--seed_base", type=int, default=SEED_BASE)
    parser.add_argument("--output_dir", default=RESULT_DIR)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Torch device: auto (default), cpu, or cuda")
    parser.add_argument("--trace", action="store_true",
                        help="Collect trace rows for node selection/evaluation behavior")
    parser.add_argument("--trace_output", default=TRACE_OUTPUT,
                        help="Path for exp22 trace CSV")
    parser.add_argument("--trace_summary", default=TRACE_SUMMARY,
                        help="Path for exp22 trace summary text")
    args = parser.parse_args()

    global DEVICE
    DEVICE = _resolve_device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = QuantumCodeRegistry.get_code(CODE_NAME)
    num_qubits = _extract_num_qubits(cfg)
    x_stabs, z_stabs = split_stabs(cfg["stabs"])
    if not x_stabs or not z_stabs:
        raise ValueError(f"{CODE_NAME} is not a pure CSS code with X and Z stabilizers.")

    topo = get_all_to_all_edges(num_qubits)
    solver = HeuristicRolloutSolver(topo, num_qubits, code_name=CODE_NAME)
    verifier = _verifier_for(cfg)
    env = SurfaceEnv(x_stabs, num_qubits, topo)

    value_cfg = _load_value_network(args.value_ckpt)
    model, normalizer, model_type, max_rank = value_cfg
    value_cfg = (model, normalizer, model_type, max_rank)

    all_rows: List[ArmRun] = []
    all_convergence: List[Dict] = []
    trace_recorder = TraceRecorder(args.trace_output) if args.trace else None

    print("=" * 57)
    print("Experiment 22")
    print("Does Value Network Improve MCTS?")
    print("=" * 57)
    print(f"Code: {CODE_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Runs: {args.runs}")
    print(f"Iterations: {args.iterations}")
    print(f"Rollouts: {args.rollouts}")
    print(f"Value checkpoint: {args.value_ckpt}")

    for i in range(args.runs):
        seed = args.seed_base + i
        t_run = time.perf_counter()
        print(f"\n--- Run {i+1}/{args.runs}  seed={seed} ---", flush=True)
        baseline_row, baseline_curve = run_arm(
            "Baseline", seed, env, solver, verifier,
            iterations=args.iterations, rollouts=args.rollouts,
            record_interval=args.record_interval, verify_interval=args.verify_interval,
            value_cfg=None,
            trace_recorder=trace_recorder,
            run_id=i,
        )
        value_row, value_curve = run_arm(
            "ValueNet", seed, env, solver, verifier,
            iterations=args.iterations, rollouts=args.rollouts,
            record_interval=args.record_interval, verify_interval=args.verify_interval,
            value_cfg=value_cfg,
            trace_recorder=trace_recorder,
            run_id=i,
        )
        all_rows.extend([baseline_row, value_row])
        all_convergence.extend(baseline_curve)
        all_convergence.extend(value_curve)

    baseline_rows = [r for r in all_rows if r.arm == "Baseline"]
    value_rows = [r for r in all_rows if r.arm == "ValueNet"]

    baseline_summary = _summarise_arm(baseline_rows)
    value_summary = _summarise_arm(value_rows)

    if trace_recorder is not None:
        trace_recorder.write_csv(args.trace_output)
        trace_recorder.write_summary(args.trace_summary)

    paired_cnot = _paired_stats([r.cnot for r in baseline_rows], [r.cnot for r in value_rows])
    paired_depth = _paired_stats([r.depth for r in baseline_rows], [r.depth for r in value_rows])
    paired_runtime = _paired_stats([r.runtime_s for r in baseline_rows], [r.runtime_s for r in value_rows])

    valid_rate = min(baseline_summary["valid_rate"], value_summary["valid_rate"])
    raw_rows = [
        {
            "seed": r.seed,
            "arm": r.arm,
            "cnot": r.cnot,
            "depth": r.depth,
            "runtime_s": r.runtime_s,
            "is_valid": r.is_valid,
            "logical_ok": r.logical_ok,
            "x_syndrome": r.x_syndrome,
            "z_syndrome": r.z_syndrome,
            "syndrome_error": r.syndrome_error,
            "first_valid_iteration": r.first_valid_iteration,
            "first_best_iteration": r.first_best_iteration,
            "best_iteration": r.best_iteration,
            "search_time": r.search_time,
            "value_inference_time": r.value_inference_time,
            "simulation_time": r.simulation_time,
            "node_count": r.node_count,
        }
        for r in all_rows
    ]

    stats_rows = [
        {
            "metric": "cnot",
            "baseline_mean": baseline_summary["cnot_mean"],
            "baseline_std": baseline_summary["cnot_std"],
            "baseline_ci95_low": _ci95([r.cnot for r in baseline_rows])[0],
            "baseline_ci95_high": _ci95([r.cnot for r in baseline_rows])[1],
            "value_mean": value_summary["cnot_mean"],
            "value_std": value_summary["cnot_std"],
            "value_ci95_low": _ci95([r.cnot for r in value_rows])[0],
            "value_ci95_high": _ci95([r.cnot for r in value_rows])[1],
            **paired_cnot,
        },
        {
            "metric": "depth",
            "baseline_mean": baseline_summary["depth_mean"],
            "baseline_std": baseline_summary["depth_std"],
            "baseline_ci95_low": _ci95([r.depth for r in baseline_rows])[0],
            "baseline_ci95_high": _ci95([r.depth for r in baseline_rows])[1],
            "value_mean": value_summary["depth_mean"],
            "value_std": value_summary["depth_std"],
            "value_ci95_low": _ci95([r.depth for r in value_rows])[0],
            "value_ci95_high": _ci95([r.depth for r in value_rows])[1],
            **paired_depth,
        },
        {
            "metric": "runtime_s",
            "baseline_mean": baseline_summary["runtime_mean"],
            "baseline_std": baseline_summary["runtime_std"],
            "baseline_ci95_low": _ci95([r.runtime_s for r in baseline_rows])[0],
            "baseline_ci95_high": _ci95([r.runtime_s for r in baseline_rows])[1],
            "value_mean": value_summary["runtime_mean"],
            "value_std": value_summary["runtime_std"],
            "value_ci95_low": _ci95([r.runtime_s for r in value_rows])[0],
            "value_ci95_high": _ci95([r.runtime_s for r in value_rows])[1],
            **paired_runtime,
        },
    ]

    convergence_rows = sorted(
        all_convergence,
        key=lambda r: (r["arm"], r["seed"], r["iteration"]),
    )

    raw_path = os.path.join(args.output_dir, "exp22_raw.csv")
    stats_path = os.path.join(args.output_dir, "exp22_statistics.csv")
    conv_path = os.path.join(args.output_dir, "exp22_convergence.csv")
    summary_path = os.path.join(args.output_dir, "exp22_summary.txt")
    plot_path = os.path.join(args.output_dir, "exp22_convergence.png")

    _write_csv(raw_path, raw_rows, [
        "seed", "arm", "cnot", "depth", "runtime_s", "is_valid", "logical_ok",
        "x_syndrome", "z_syndrome", "syndrome_error", "first_valid_iteration",
        "first_best_iteration", "best_iteration", "search_time",
        "value_inference_time", "simulation_time", "node_count",
    ])
    _write_csv(stats_path, stats_rows, [
        "metric",
        "baseline_mean", "baseline_std", "baseline_ci95_low", "baseline_ci95_high",
        "value_mean", "value_std", "value_ci95_low", "value_ci95_high",
        "delta_mean", "delta_std", "ci95_low", "ci95_high",
        "paired_t", "paired_t_p", "wilcoxon", "wilcoxon_p", "cohen_dz",
    ])
    _write_csv(conv_path, convergence_rows, [
        "seed", "arm", "iteration", "best_cnot", "best_depth", "is_valid",
        "logical_ok", "x_syndrome", "z_syndrome", "runtime_s",
        "first_valid_iteration", "first_best_iteration",
    ])

    _write_summary(
        summary_path,
        args,
        baseline_summary,
        value_summary,
        paired_cnot,
        paired_depth,
        paired_runtime,
        valid_rate,
    )
    _maybe_plot(convergence_rows, plot_path)

    print(f"Saved: {raw_path}")
    print(f"Saved: {stats_path}")
    print(f"Saved: {conv_path}")
    print(f"Saved: {summary_path}")


def _write_summary(path, args, baseline_summary, value_summary,
                   paired_cnot, paired_depth, paired_runtime, valid_rate):
    lines = []
    lines.append("=================================================")
    lines.append("Experiment 22")
    lines.append("Does Value Network Improve MCTS?")
    lines.append("=================================================")
    lines.append("")
    lines.append(f"Runs: {args.runs}")
    lines.append(f"Iterations: {args.iterations}")
    lines.append(f"Rollouts: {args.rollouts}")
    lines.append(f"Code: {CODE_NAME}")
    lines.append("")
    lines.append("-----------------------------------------------")
    lines.append("Baseline")
    lines.append(f"CNOT: {baseline_summary['cnot_mean']:.2f} +/- {baseline_summary['cnot_std']:.2f}")
    lines.append(f"Depth: {baseline_summary['depth_mean']:.2f} +/- {baseline_summary['depth_std']:.2f}")
    lines.append(f"Time: {baseline_summary['runtime_mean']:.2f} +/- {baseline_summary['runtime_std']:.2f} s")
    lines.append(f"Valid Rate: {baseline_summary['valid_rate']*100:.1f}%")
    lines.append("-----------------------------------------------")
    lines.append("ValueNet")
    lines.append(f"CNOT: {value_summary['cnot_mean']:.2f} +/- {value_summary['cnot_std']:.2f}")
    lines.append(f"Depth: {value_summary['depth_mean']:.2f} +/- {value_summary['depth_std']:.2f}")
    lines.append(f"Time: {value_summary['runtime_mean']:.2f} +/- {value_summary['runtime_std']:.2f} s")
    lines.append(f"Valid Rate: {value_summary['valid_rate']*100:.1f}%")
    lines.append("-----------------------------------------------")
    lines.append("Difference (Value - Baseline)")
    lines.append(f"Delta CNOT: {paired_cnot['delta_mean']:.2f}")
    lines.append(f"Delta Depth: {paired_depth['delta_mean']:.2f}")
    lines.append(f"Delta Time: {paired_runtime['delta_mean']:.2f} s")
    lines.append(f"paired t-test p (CNOT): {paired_cnot['paired_t_p']:.6g}")
    lines.append(f"Wilcoxon p (CNOT): {paired_cnot['wilcoxon_p']:.6g}")
    lines.append(f"Effect size dz (CNOT): {paired_cnot['cohen_dz']:.4f}")
    lines.append(f"Valid Rate floor: {valid_rate*100:.1f}%")
    lines.append("-----------------------------------------------")
    if valid_rate < 1.0:
        lines.append("Conclusion")
        lines.append("Data invalid because valid_rate < 100%.")
    else:
        better = "improves" if paired_cnot["delta_mean"] < 0 else "does not improve"
        lines.append("Conclusion")
        lines.append(f"Value Network {better} MCTS under the current data.")
    lines.append("=================================================")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _maybe_plot(convergence_rows, plot_path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not convergence_rows:
        return

    groups: Dict[Tuple[str, int], List[Dict]] = {}
    for row in convergence_rows:
        groups.setdefault((row["arm"], row["seed"]), []).append(row)

    plt.figure(figsize=(9, 5))
    for arm in ["Baseline", "ValueNet"]:
        arm_groups = [rows for (a, _), rows in groups.items() if a == arm]
        if not arm_groups:
            continue
        iterations = sorted({r["iteration"] for rows in arm_groups for r in rows})
        mean_curve = []
        for it in iterations:
            vals = [r["best_cnot"] for rows in arm_groups for r in rows if r["iteration"] == it]
            mean_curve.append(float(np.mean(vals)) if vals else float("nan"))
        plt.plot(iterations, mean_curve, marker="o", linewidth=2, label=arm)

    plt.xlabel("Iteration")
    plt.ylabel("Best CNOT so far")
    plt.title("Exp22 Convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
