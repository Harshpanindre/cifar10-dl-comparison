"""Common evaluation protocol: accuracy, macro precision / recall / F1, confusion matrix."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)


def compute_metrics(y_true, y_pred, class_names):
    labels = list(range(len(class_names)))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    pc_p, pc_r, pc_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
        "per_class": {
            name: {"precision": float(pc_p[i]), "recall": float(pc_r[i]),
                   "f1": float(pc_f1[i]), "support": int(support[i])}
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
    }


def plot_confusion_matrix(cm, class_names, path, title="Confusion matrix", normalize=True):
    cm = np.asarray(cm, dtype=float)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, val, ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > 0.5 * cm.max() else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history, path, title):
    epochs = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(epochs, [h["train_loss"] for h in history], label="train")
    a1.plot(epochs, [h["val_loss"] for h in history], label="val")
    a1.set_xlabel("epoch"); a1.set_ylabel("loss"); a1.legend(); a1.set_title("Loss")
    a2.plot(epochs, [h["train_acc"] for h in history], label="train")
    a2.plot(epochs, [h["val_acc"] for h in history], label="val")
    a2.set_xlabel("epoch"); a2.set_ylabel("accuracy"); a2.legend(); a2.set_title("Accuracy")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
