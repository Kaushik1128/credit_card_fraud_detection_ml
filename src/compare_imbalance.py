"""Compare four imbalance-handling strategies on XGBoost.

Strategies:
  1. none              — XGBoost as-is, trust the trees
  2. scale_pos_weight  — multiply fraud gradient by n_neg/n_pos (~578)
  3. SMOTE             — synthesize fraud rows up to 50/50, then train
  4. undersample       — drop genuine rows down to ~50/50, then train

Run from project root:
    .venv/Scripts/python.exe -m src.compare_imbalance
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data import RANDOM_STATE, load_splits, make_preprocessor, split_xy
from src.metrics import evaluate

OUT_PATH = Path("models/imbalance_comparison_metrics.json")
FIG_PATH = Path("notebooks/figures/12_imbalance_pr_comparison.png")


def base_classifier(scale_pos_weight: float = 1.0) -> XGBClassifier:
    """Same hyperparameters as Phase 2.4 — only scale_pos_weight varies per strategy."""
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_strategies(spw: float) -> dict[str, Pipeline]:
    """Each value is a fittable pipeline; SMOTE/undersample use imblearn.Pipeline
    so the resampler is skipped at predict time."""
    return {
        "none": Pipeline([
            ("pre", make_preprocessor()),
            ("clf", base_classifier()),
        ]),
        "scale_pos_weight": Pipeline([
            ("pre", make_preprocessor()),
            ("clf", base_classifier(scale_pos_weight=spw)),
        ]),
        "SMOTE": ImbPipeline([
            ("pre", make_preprocessor()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", base_classifier()),
        ]),
        "undersample": ImbPipeline([
            ("pre", make_preprocessor()),
            ("under", RandomUnderSampler(random_state=RANDOM_STATE)),
            ("clf", base_classifier()),
        ]),
    }


def main() -> None:
    splits = load_splits()
    X_train, y_train = split_xy(splits["train"])
    X_val, y_val = split_xy(splits["val"])

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw = n_neg / n_pos
    print(f"train = {len(X_train):,}  fraud = {n_pos}  scale_pos_weight = {spw:.1f}")

    strategies = build_strategies(spw)

    results: dict[str, dict] = {}
    pr_curves: dict[str, tuple] = {}

    for name, pipe in strategies.items():
        print(f"\n=== {name} ===  fitting...")
        pipe.fit(X_train, y_train)
        m = evaluate(pipe, X_val, y_val, threshold=0.5)
        results[name] = m
        y_score = pipe.predict_proba(X_val)[:, 1]
        pr_curves[name] = (y_val.values, y_score)
        cm = m["confusion_matrix"]
        print(
            f"  PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}  "
            f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
            f"FP={cm['fp']}  FN={cm['fn']}"
        )

    # Comparison table
    print("\n" + "=" * 72)
    print("Summary (validation):")
    print(f"  {'strategy':<20} {'PR-AUC':>8} {'ROC-AUC':>8} {'Prec':>6} {'Recall':>7} {'F1':>6}  {'FP':>5}  {'FN':>4}")
    for name, m in results.items():
        cm = m["confusion_matrix"]
        print(
            f"  {name:<20} {m['pr_auc']:>8.4f} {m['roc_auc']:>8.4f} "
            f"{m['precision']:>6.3f} {m['recall']:>7.3f} {m['f1']:>6.3f}  "
            f"{cm['fp']:>5}  {cm['fn']:>4}"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics -> {OUT_PATH}")

    # Overlay PR plot
    colors = {
        "none":             "#95a5a6",
        "scale_pos_weight": "#9b59b6",
        "SMOTE":            "#16a085",
        "undersample":      "#e67e22",
    }
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (yt, ys) in pr_curves.items():
        p, r, _ = precision_recall_curve(yt, ys)
        ap = average_precision_score(yt, ys)
        ax.plot(r, p, label=f"{name}  (AP = {ap:.3f})", color=colors[name], lw=2)
    baseline = float(y_val.mean())
    ax.axhline(baseline, ls="--", color="gray", lw=1, label=f"random (AP = {baseline:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("XGBoost imbalance strategies (val)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    plt.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_PATH, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved plot    -> {FIG_PATH}")


if __name__ == "__main__":
    main()
