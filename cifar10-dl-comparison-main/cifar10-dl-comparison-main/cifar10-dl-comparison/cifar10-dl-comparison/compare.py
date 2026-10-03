"""Build the single comparison table across all trained models.

    python compare.py
Writes results/comparison.md, comparison.csv, comparison.png, per_class_f1.csv, per_class_f1.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import CLASSES
from src.utils import load_config

ORDER = ["mlp", "cnn", "lstm", "vit"]


def main():
    out = Path(load_config("mlp")["output_dir"])
    rows, per_class = [], {}
    for name in ORDER:
        mpath, spath = out / name / "test_metrics.json", out / name / "train_summary.json"
        if not mpath.exists():
            print(f"skip {name}: no test_metrics.json (run train.py and evaluate.py first)")
            continue
        m = json.loads(mpath.read_text())
        s = json.loads(spath.read_text()) if spath.exists() else {}
        rows.append({
            "Model": m["display_name"],
            "Params (M)": round(s.get("params", float("nan")) / 1e6, 2),
            "Epochs": s.get("epochs_run"),
            "Train time (min)": round(s.get("train_time_min", float("nan")), 1),
            "Best val acc (%)": round(100 * s.get("best_val_acc", float("nan")), 2),
            "Test acc (%)": round(100 * m["accuracy"], 2),
            "Precision": round(m["precision_macro"], 4),
            "Recall": round(m["recall_macro"], 4),
            "F1 (macro)": round(m["f1_macro"], 4),
            "Inference (ms/img)": round(m["inference_ms_per_image"], 3),
        })
        per_class[m["display_name"]] = [m["per_class"][c]["f1"] for c in CLASSES]

    if not rows:
        print("Nothing to compare yet.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out / "comparison.csv", index=False)
    md = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    (out / "comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    pc = pd.DataFrame(per_class, index=CLASSES)
    pc.round(4).to_csv(out / "per_class_f1.csv")

    # bar chart of the common metrics
    metrics = ["Test acc (%)", "Precision", "Recall", "F1 (macro)"]
    vals = df[metrics].to_numpy(dtype=float)
    vals[:, 0] /= 100
    x = np.arange(len(metrics))
    width = 0.8 / len(df)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, name in enumerate(df["Model"]):
        ax.bar(x + i * width - 0.4 + width / 2, vals[i], width, label=name)
    ax.set_xticks(x, ["Accuracy", "Precision", "Recall", "F1 (macro)"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("score (test set)")
    ax.legend(fontsize=8)
    ax.set_title("CIFAR-10 test performance by architecture family")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=150)
    plt.close(fig)

    # per-class F1 heatmap: shows WHICH classes each family struggles with
    fig, ax = plt.subplots(figsize=(1.6 * len(pc.columns) + 2, 5))
    im = ax.imshow(pc.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pc.columns)), pc.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(pc.columns)):
            ax.text(j, i, f"{pc.iat[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Per-class F1 (test set)")
    fig.tight_layout()
    fig.savefig(out / "per_class_f1.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved comparison files to {out}/")


if __name__ == "__main__":
    main()
