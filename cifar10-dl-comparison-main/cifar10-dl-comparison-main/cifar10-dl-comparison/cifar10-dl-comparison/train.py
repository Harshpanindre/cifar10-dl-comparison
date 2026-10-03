"""Train one model under the common protocol and keep the best checkpoint (by validation accuracy).

Examples:
    python train.py --model cnn
    python train.py --model vit --set train.epochs=100 --set optim.lr=0.0005
    python train.py --model lstm --set model.sequence=patches
    python train.py --model mlp --fake-data --epochs 2 --max-batches 5     # quick pipeline check
"""
import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.data import NUM_CLASSES, get_loaders
from src.engine import evaluate, train_one_epoch
from src.metrics import plot_history
from src.models import MODEL_REGISTRY, DISPLAY_NAMES, build_model
from src.utils import (build_optimizer, build_scheduler, count_parameters, get_device,
                       load_config, save_json, set_seed)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    ap.add_argument("--config", default=None, help="defaults to configs/<model>.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="override any config value, e.g. --set train.epochs=10")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--run-name", default=None, help="output sub-folder (default: model name)")
    ap.add_argument("--fake-data", action="store_true", help="random images, no download (pipeline test)")
    ap.add_argument("--max-batches", type=int, default=None, help="limit batches per epoch (debugging)")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.model, args.config, args.overrides)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    epochs = int(cfg["train"]["epochs"])

    set_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    device = get_device()
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"

    run_name = args.run_name or cfg["model"]["name"]
    out_dir = Path(cfg["output_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, out_dir / "config_used.json")

    train_loader, val_loader, _ = get_loaders(cfg["data"], cfg["seed"], fake=args.fake_data)
    model = build_model(cfg["model"], NUM_CLASSES).to(device)
    n_params = count_parameters(model)
    print(f"Model: {DISPLAY_NAMES[cfg['model']['name']]} | params: {n_params / 1e6:.2f}M | device: {device} | AMP: {use_amp}")

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["train"].get("label_smoothing", 0.0)))
    optimizer = build_optimizer(model, cfg["optim"])
    steps = len(train_loader) if not args.max_batches else min(len(train_loader), args.max_batches)
    scheduler = build_scheduler(optimizer, cfg.get("scheduler"), steps, epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    grad_clip = float(cfg["train"].get("grad_clip", 0.0))
    patience = int(cfg["train"].get("early_stopping_patience", 0))

    history, best_acc, best_epoch, bad_epochs = [], -1.0, 0, 0
    hist_path = out_dir / "history.csv"
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler,
                             scheduler, use_amp, grad_clip, args.max_batches)
        va = evaluate(model, val_loader, criterion, device, use_amp, args.max_batches)
        row = {"epoch": epoch, "train_loss": tr["loss"], "train_acc": tr["acc"],
               "val_loss": va["loss"], "val_acc": va["acc"],
               "lr": optimizer.param_groups[0]["lr"], "epoch_time_s": time.time() - t0}
        history.append(row)
        print(f"[{epoch:3d}/{epochs}] train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
              f"val loss {va['loss']:.4f} acc {va['acc']:.4f} | lr {row['lr']:.2e} | {row['epoch_time_s']:.1f}s")

        with open(hist_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(history)

        if va["acc"] > best_acc:
            best_acc, best_epoch, bad_epochs = va["acc"], epoch, 0
            torch.save({"model_state": model.state_dict(), "config": cfg,
                        "epoch": epoch, "val_acc": float(best_acc)}, out_dir / "best.pt")
        else:
            bad_epochs += 1
            if patience and bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    train_time = time.time() - t_start
    summary = {
        "model": cfg["model"]["name"],
        "display_name": DISPLAY_NAMES[cfg["model"]["name"]],
        "params": n_params,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "train_time_min": train_time / 60,
        "device": str(device),
        "fake_data": args.fake_data,
    }
    save_json(summary, out_dir / "train_summary.json")
    plot_history(history, out_dir / "training_curves.png", summary["display_name"])
    print(f"Done. Best val acc {best_acc:.4f} at epoch {best_epoch}. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
