"""Evaluate the best checkpoint of a model on the held-out TEST set (run once per final model).

    python evaluate.py --model cnn
Writes results/<model>/test_metrics.json, confusion_matrix.png and predictions.npz
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data import CLASSES, NUM_CLASSES, get_loaders
from src.engine import evaluate
from src.metrics import compute_metrics, plot_confusion_matrix
from src.models import MODEL_REGISTRY, DISPLAY_NAMES, build_model
from src.utils import get_device, load_config, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--fake-data", action="store_true")
    args = ap.parse_args()

    base_cfg = load_config(args.model)
    run_dir = Path(base_cfg["output_dir"]) / (args.run_name or args.model)
    device = get_device()

    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    cfg = ckpt["config"]                      # exact config the model was trained with
    model = build_model(cfg["model"], NUM_CLASSES).to(device)
    model.load_state_dict(ckpt["model_state"])

    _, _, test_loader = get_loaders(cfg["data"], cfg["seed"], fake=args.fake_data)
    criterion = nn.CrossEntropyLoss()

    t0 = time.time()
    res = evaluate(model, test_loader, criterion, device, use_amp=False)
    elapsed = time.time() - t0

    metrics = compute_metrics(res["y_true"], res["y_pred"], CLASSES)
    metrics.update({
        "model": args.model,
        "display_name": DISPLAY_NAMES[args.model],
        "test_loss": res["loss"],
        "checkpoint_epoch": ckpt["epoch"],
        "checkpoint_val_acc": ckpt["val_acc"],
        "inference_ms_per_image": 1000 * elapsed / len(res["y_true"]),
    })
    save_json(metrics, run_dir / "test_metrics.json")
    plot_confusion_matrix(metrics["confusion_matrix"], CLASSES, run_dir / "confusion_matrix.png",
                          title=f"{DISPLAY_NAMES[args.model]} — test set (row-normalised)")
    np.savez_compressed(run_dir / "predictions.npz", y_true=res["y_true"],
                        y_pred=res["y_pred"], probs=res["probs"])

    print(f"{DISPLAY_NAMES[args.model]} TEST | acc {metrics['accuracy']:.4f} | "
          f"P {metrics['precision_macro']:.4f} | R {metrics['recall_macro']:.4f} | "
          f"F1 {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
