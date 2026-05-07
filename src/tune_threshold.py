"""Tune the decision threshold of the trained XGBoost model.

The model is fixed — we only choose where on its precision/recall curve
to operate. Sweep thresholds on val and surface four candidates:

  - Default (t = 0.5)
  - Best F1
  - Smallest threshold with recall  >= 0.95
  - Smallest threshold with precision >= 0.95
  - Cost-minimizing under FN=$100, FP=$10  (winner candidate)

Run from project root:
    .venv/Scripts/python.exe -m src.tune_threshold
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from src.data import load_splits, split_xy

MODEL_PATH = Path("models/xgboost_baseline.joblib")
OUT_PATH = Path("models/xgboost_threshold_analysis.json")
FIG_PATH = Path("notebooks/figures/13_threshold_curves.png")

# Illustrative cost numbers. Real costs come from the bank's ops + risk team.
COST_FN = 100.0   # average dollar value of a missed fraud
COST_FP = 10.0    # cost of a customer-friction call for a false alarm

RECALL_TARGET = 0.95
PRECISION_TARGET = 0.95


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, t: float) -> dict:
    y_pred = (y_score >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0  # convention: no alerts -> "perfect" precision
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    cost = fn * COST_FN + fp * COST_FP
    return {
        "threshold": float(t),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "expected_cost": float(cost),
    }


def main() -> None:
    print("Loading model and val split...")
    model = joblib.load(MODEL_PATH)
    splits = load_splits()
    X_val, y_val = split_xy(splits["val"])
    y_true = y_val.values
    y_score = model.predict_proba(X_val)[:, 1]

    print(f"  val rows = {len(y_true):,}  fraud = {int(y_true.sum())}")
    print(f"  cost model: FN = ${COST_FN:.0f}, FP = ${COST_FP:.0f}")

    # Sweep 999 thresholds on (0, 1).
    thresholds = np.linspace(0.001, 0.999, 999)
    rows = [metrics_at_threshold(y_true, y_score, t) for t in thresholds]

    # Candidate thresholds.
    default_row   = metrics_at_threshold(y_true, y_score, 0.5)
    best_f1_row   = max(rows, key=lambda r: r["f1"])
    cost_min_row  = min(rows, key=lambda r: r["expected_cost"])
    recall_95_row = next((r for r in rows if r["recall"] >= RECALL_TARGET), None)
    # for precision target we want the SMALLEST threshold that still hits it
    # (smallest t -> highest recall while satisfying precision target)
    prec_target_rows = [r for r in rows if r["precision"] >= PRECISION_TARGET]
    precision_95_row = min(prec_target_rows, key=lambda r: r["threshold"]) if prec_target_rows else None

    candidates = {
        "default_0.5":   default_row,
        "best_f1":       best_f1_row,
        "recall_>=0.95": recall_95_row,
        "precision_>=0.95": precision_95_row,
        "cost_min":      cost_min_row,
    }

    # Pretty-print comparison.
    print("\n" + "=" * 88)
    print(f"  {'candidate':<18}  {'thresh':>6} {'P':>6} {'R':>6} {'F1':>6}  "
          f"{'TP':>3} {'FP':>5} {'FN':>4}  {'cost':>10}")
    print("-" * 88)
    for label, r in candidates.items():
        if r is None:
            print(f"  {label:<18}  -- not achievable --")
            continue
        print(f"  {label:<18}  {r['threshold']:>6.3f} {r['precision']:>6.3f} "
              f"{r['recall']:>6.3f} {r['f1']:>6.3f}  "
              f"{r['tp']:>3} {r['fp']:>5} {r['fn']:>4}  ${r['expected_cost']:>8,.2f}")

    # Persist analysis.
    summary = {
        "cost_model": {"fn": COST_FN, "fp": COST_FP},
        "recall_target": RECALL_TARGET,
        "precision_target": PRECISION_TARGET,
        "candidates": candidates,
        "all_thresholds": rows,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved analysis -> {OUT_PATH}")

    # Plot P/R/F1 vs threshold + cost on a secondary axis.
    ts = [r["threshold"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(ts, [r["precision"] for r in rows], color="#3498db", lw=2, label="Precision")
    ax1.plot(ts, [r["recall"]    for r in rows], color="#e74c3c", lw=2, label="Recall")
    ax1.plot(ts, [r["f1"]        for r in rows], color="#27ae60", lw=2, label="F1")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Precision / Recall / F1")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.02)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(ts, [r["expected_cost"] for r in rows], color="#7f8c8d", ls="--", lw=1.5,
             label=f"Expected $ cost (FN=${COST_FN:.0f}, FP=${COST_FP:.0f})")
    ax2.set_ylabel("Expected cost ($)")

    # Annotate the candidate thresholds.
    for label, r, color in [
        ("default 0.5", default_row, "gray"),
        ("best F1", best_f1_row, "#27ae60"),
        ("cost min", cost_min_row, "#7f8c8d"),
    ]:
        if r is None:
            continue
        ax1.axvline(r["threshold"], ls=":", alpha=0.6, color=color)
        ax1.text(r["threshold"] + 0.005, 0.05, f"{label}\nt={r['threshold']:.2f}",
                 fontsize=8, color=color)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title("XGBoost: precision / recall / F1 / cost vs threshold (val)")
    plt.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_PATH, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved plot     -> {FIG_PATH}")


if __name__ == "__main__":
    main()
