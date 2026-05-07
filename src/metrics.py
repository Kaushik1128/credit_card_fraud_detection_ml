"""Evaluation utilities for binary fraud classifiers.

Pure functions — pass in a fitted model + (X, y), get back a metrics dict
and saved plots. Reused across all model phases (LR, XGBoost, imbalance experiments).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive: just write PNGs, never pop a window

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate(model, X, y, threshold: float = 0.5) -> dict[str, Any]:
    """Score a fitted model on (X, y).

    Returns a dict with all panel metrics. JSON-serializable so it can be
    written to disk and diffed in git.
    """
    y_score = model.predict_proba(X)[:, 1]
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return {
        "n_samples": int(len(y)),
        "n_positives": int(y.sum()),
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y, y_score)),
        "roc_auc": float(roc_auc_score(y, y_score)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def print_report(metrics: dict[str, Any], title: str = "Evaluation") -> None:
    cm = metrics["confusion_matrix"]
    print(f"\n{title}  (n={metrics['n_samples']:,}, fraud={metrics['n_positives']:,})")
    print("-" * 60)
    print(f"  PR-AUC      = {metrics['pr_auc']:.4f}   (primary metric)")
    print(f"  ROC-AUC     = {metrics['roc_auc']:.4f}")
    print(f"  Precision   = {metrics['precision']:.4f}   @ threshold {metrics['threshold']}")
    print(f"  Recall      = {metrics['recall']:.4f}")
    print(f"  F1          = {metrics['f1']:.4f}")
    print(f"  Confusion matrix:")
    print(f"                  pred 0       pred 1")
    print(f"     actual 0   {cm['tn']:>9,}    {cm['fp']:>9,}")
    print(f"     actual 1   {cm['fn']:>9,}    {cm['tp']:>9,}")


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def plot_confusion_matrix(y_true, y_pred, save_path: str | Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["genuine", "fraud"])
    ax.set_yticklabels(["genuine", "fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    threshold = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=14, fontweight="bold",
            )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_pr_curve(y_true, y_score, save_path: str | Path, label: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    baseline = float(np.mean(y_true))  # precision a random ranker would get

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"{label} (AP = {ap:.3f})", color="#3498db", lw=2)
    ax.axhline(baseline, ls="--", color="gray", lw=1, label=f"random (AP = {baseline:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_score, save_path: str | Path, label: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})", color="#e74c3c", lw=2)
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
