"""Error analysis for one model: most-confused class pairs and the most confident mistakes.

    python error_analysis.py --model cnn
Writes results/<model>/confused_pairs.csv and misclassified.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import CLASSES, get_raw_test_images
from src.models import DISPLAY_NAMES, MODEL_REGISTRY
from src.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--top-pairs", type=int, default=10)
    ap.add_argument("--n-images", type=int, default=24)
    ap.add_argument("--fake-data", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.model)
    run_dir = Path(cfg["output_dir"]) / (args.run_name or args.model)
    pred = np.load(run_dir / "predictions.npz")
    y_true, y_pred, probs = pred["y_true"], pred["y_pred"], pred["probs"]

    # 1) most confused (true -> predicted) pairs
    k = len(CLASSES)
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    np.fill_diagonal(cm, 0)
    pairs = [(CLASSES[i], CLASSES[j], int(cm[i, j])) for i in range(k) for j in range(k) if cm[i, j] > 0]
    pairs.sort(key=lambda t: -t[2])
    df = pd.DataFrame(pairs[:args.top_pairs], columns=["true", "predicted", "count"])
    df.to_csv(run_dir / "confused_pairs.csv", index=False)
    print(df.to_string(index=False))

    # 2) most confident mistakes (the model was "sure" and wrong)
    wrong = np.where(y_true != y_pred)[0]
    conf = probs[wrong, y_pred[wrong]]
    worst = wrong[np.argsort(-conf)][:args.n_images]
    images = get_raw_test_images(cfg["data"], fake=args.fake_data)

    cols = 6
    rows = int(np.ceil(len(worst) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.2))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, idx in zip(np.atleast_1d(axes).ravel(), worst):
        ax.imshow(images[idx])
        ax.set_title(f"T: {CLASSES[y_true[idx]]}\nP: {CLASSES[y_pred[idx]]} ({probs[idx, y_pred[idx]]:.2f})",
                     fontsize=7)
    fig.suptitle(f"{DISPLAY_NAMES[args.model]} — most confident misclassifications")
    fig.tight_layout()
    fig.savefig(run_dir / "misclassified.png", dpi=150)
    plt.close(fig)
    print(f"Saved {run_dir}/confused_pairs.csv and misclassified.png")


if __name__ == "__main__":
    main()
