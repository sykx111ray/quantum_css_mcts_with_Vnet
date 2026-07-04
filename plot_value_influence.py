import csv
import os
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
TRACE_CSV = os.path.join(ROOT, "results", "exp22_trace.csv")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_trace(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _to_float(v):
    if v in (None, ""):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def plot_entropy(rows):
    by_mode = defaultdict(list)
    for r in rows:
        mode = r["mode"]
        step = int(r["step"])
        ent = _to_float(r["visit_entropy"])
        if ent is None:
            continue
        by_mode[mode].append((step, ent))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for mode in ["baseline", "value"]:
        pts = sorted(by_mode.get(mode, []))
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if mode == "baseline":
            ax.plot(xs, ys, label="Baseline", color="tab:blue", linewidth=1.8)
        else:
            ax.plot(xs, ys, label="ValueNet", color="tab:orange", linewidth=1.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Visit Entropy")
    ax.set_title("Node Visit Entropy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "entropy_curve.png"), dpi=200)
    plt.close(fig)


def plot_actions(rows):
    by_mode = defaultdict(list)
    for r in rows:
        by_mode[r["mode"]].append(r["selected_action"])

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for mode in ["baseline", "value"]:
        counts = defaultdict(int)
        for action in by_mode.get(mode, []):
            if action:
                counts[action] += 1
        items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:20]
        labels = [a for a, _ in items]
        values = [c for _, c in items]
        if mode == "baseline":
            ax.bar(labels, values, alpha=0.6, label="Baseline", color="tab:blue")
        else:
            ax.bar(labels, values, alpha=0.6, label="ValueNet", color="tab:orange")
    ax.set_title("Selected Action Distribution")
    ax.set_xlabel("Action")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "action_distribution.png"), dpi=200)
    plt.close(fig)


def plot_value_vs_rollout(rows):
    vals = []
    for r in rows:
        if r["mode"] != "value":
            continue
        vp = _to_float(r["value_pred"])
        rm = _to_float(r["rollout_min"])
        if vp is None or rm is None:
            continue
        vals.append((vp, rm))
    if not vals:
        return
    vps = [v[0] for v in vals]
    rms = [v[1] for v in vals]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
    axs[0].scatter(rms, vps, alpha=0.5, s=14)
    axs[0].plot([min(rms), max(rms)], [min(rms), max(rms)], linestyle="--", color="gray")
    axs[0].set_xlabel("Rollout Min")
    axs[0].set_ylabel("ValueNet Prediction")
    axs[0].set_title("Value vs Rollout Min")
    axs[0].grid(True, alpha=0.25)

    zscores = []
    for r in rows:
        if r["mode"] != "value":
            continue
        z = _to_float(r["value_zscore"])
        if z is not None:
            zscores.append(z)
    if zscores:
        axs[1].hist(zscores, bins=30, color="tab:green", alpha=0.7)
        axs[1].set_xlabel("Value Z-score")
        axs[1].set_ylabel("Count")
        axs[1].set_title("Value vs Rollout Z-score")
        axs[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "value_vs_rollout.png"), dpi=200)
    plt.close(fig)


def main():
    rows = load_trace(TRACE_CSV)
    plot_entropy(rows)
    plot_actions(rows)
    plot_value_vs_rollout(rows)


if __name__ == "__main__":
    main()
