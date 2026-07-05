"""
Final-validation helpers for MCTS/V-Net experiments.

This module is intentionally diagnostic-only: it does not choose actions,
change costs, alter backpropagation, or feed verifier output back into MCTS.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, Optional


def get_cnot_count(circuit: Optional[Iterable]) -> int:
    if not circuit:
        return 0
    return sum(1 for gate in circuit if gate and gate[0] == "CNOT")


def get_depth(circuit: Optional[Iterable]) -> int:
    return len(list(circuit)) if circuit else 0


def count_tree_nodes(root: Any) -> int:
    count = 0
    stack = [root] if root is not None else []
    while stack:
        node = stack.pop()
        count += 1
        stack.extend(getattr(node, "children", getattr(node, "ch", [])))
    return count


def aggregate_verifier_results(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    raw = [r for r in rows if _finite_nonnegative(r.get("cnot", r.get("cnot_count")))]
    valid = [r for r in raw if _as_bool(r.get("is_valid"))]

    def mean(key: str, source: Iterable[Dict[str, Any]]) -> Optional[float]:
        vals = [float(r[key]) for r in source if _is_finite(r.get(key))]
        return float(sum(vals) / len(vals)) if vals else None

    return {
        "raw_runs": len(raw),
        "valid_runs": len(valid),
        "invalid_runs": len(raw) - len(valid),
        "valid_rate": float(len(valid) / len(raw)) if raw else 0.0,
        "raw_cnot_mean": mean("cnot", raw),
        "raw_depth_mean": mean("depth", raw),
        "valid_cnot_mean": mean("cnot", valid),
        "valid_depth_mean": mean("depth", valid),
        "x_syndrome": mean("x_syndrome_error", raw),
        "z_syndrome": mean("z_syndrome_error", raw),
        "logical": _logical_summary(raw),
    }


class SearchStatsRecorder:
    """Optional per-run recorder for final validation metrics."""

    def __init__(
        self,
        csv_path: Optional[str] = None,
        record_every: int = 100,
        verifier: Optional[Callable[[Any], Dict[str, Any]]] = None,
    ) -> None:
        self.csv_path = csv_path
        self.record_every = int(record_every)
        self.verifier = verifier
        self.search_time = 0.0
        self.value_inference_time = 0.0
        self.simulation_time = 0.0
        self.first_valid_iteration: Optional[int] = None
        self.first_best_iteration: Optional[int] = None
        self._t0: Optional[float] = None
        self._best_signature: Optional[tuple] = None
        self._rows = []

    def enabled(self) -> bool:
        return bool(self.csv_path)

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def finish(self) -> None:
        if self._t0 is not None:
            self.search_time += time.perf_counter() - self._t0
            self._t0 = None
        self.flush()

    @contextmanager
    def time_value_inference(self):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.value_inference_time += time.perf_counter() - t0

    @contextmanager
    def time_simulation(self):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.simulation_time += time.perf_counter() - t0

    def observe_best(self, iteration: int, circuit: Optional[Iterable]) -> None:
        if not circuit:
            return
        sig = (get_cnot_count(circuit), get_depth(circuit))
        if sig != self._best_signature:
            self._best_signature = sig
            self.first_best_iteration = int(iteration)

    def record(self, iteration: int, circuit: Optional[Iterable], node_count: int) -> None:
        if not self.enabled() or iteration % self.record_every != 0:
            return
        cnot = get_cnot_count(circuit)
        depth = get_depth(circuit)
        is_valid = None
        x_syn = None
        z_syn = None
        logical = None
        if circuit and self.verifier is not None:
            diag = self.verifier(circuit)
            is_valid = bool(diag.get("is_valid"))
            x_syn = diag.get("x_syndrome_error")
            z_syn = diag.get("z_syndrome_error")
            logical = diag.get("is_logical_zero")
            if is_valid and self.first_valid_iteration is None:
                self.first_valid_iteration = int(iteration)

        self._rows.append({
            "iteration": int(iteration),
            "best_cnot": cnot,
            "best_depth": depth,
            "is_valid": is_valid,
            "x_syndrome_error": x_syn,
            "z_syndrome_error": z_syn,
            "is_logical_zero": logical,
            "first_valid_iteration": self.first_valid_iteration,
            "first_best_iteration": self.first_best_iteration,
            "search_time": self.elapsed_search_time(),
            "value_inference_time": self.value_inference_time,
            "simulation_time": self.simulation_time,
            "node_count": int(node_count),
        })

    def elapsed_search_time(self) -> float:
        elapsed = self.search_time
        if self._t0 is not None:
            elapsed += time.perf_counter() - self._t0
        return elapsed

    def summary(self, node_count: int) -> Dict[str, Any]:
        return {
            "first_valid_iteration": self.first_valid_iteration,
            "first_best_iteration": self.first_best_iteration,
            "search_time": self.search_time,
            "value_inference_time": self.value_inference_time,
            "simulation_time": self.simulation_time,
            "node_count": int(node_count),
        }

    def flush(self) -> None:
        if not self.enabled() or not self._rows:
            return
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        fields = list(self._rows[0].keys())
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(self._rows)


def validation_log_path(exp_name: str, run_id: int) -> Optional[str]:
    base = os.environ.get("FINAL_VALIDATION_LOG_DIR")
    if not base:
        return None
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in exp_name)
    return os.path.join(base, f"{safe}_run{run_id:03d}_anytime.csv")


def write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_nonnegative(value: Any) -> bool:
    return _is_finite(value) and float(value) >= 0


def _logical_summary(rows: Iterable[Dict[str, Any]]) -> Optional[float]:
    vals = []
    for r in rows:
        v = r.get("is_logical_zero")
        if v is None or v == "":
            continue
        vals.append(_as_bool(v))
    return float(sum(vals) / len(vals)) if vals else None
