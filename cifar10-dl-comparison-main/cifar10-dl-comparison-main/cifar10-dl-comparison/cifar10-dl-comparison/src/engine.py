"""Training and evaluation loops shared by every model."""
import numpy as np
import torch
import torch.nn as nn


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None,
                    scheduler=None, use_amp=False, grad_clip=0.0, max_batches=None):
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        seen += y.size(0)
    return {"loss": total_loss / max(1, seen), "acc": correct / max(1, seen)}


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=False, max_batches=None):
    model.eval()
    total_loss, seen = 0.0, 0
    all_true, all_pred, all_prob = [], [], []
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        probs = logits.float().softmax(1)
        total_loss += loss.item() * y.size(0)
        seen += y.size(0)
        all_true.append(y.cpu().numpy())
        all_pred.append(probs.argmax(1).cpu().numpy())
        all_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return {
        "loss": total_loss / max(1, seen),
        "acc": float((y_true == y_pred).mean()),
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": np.concatenate(all_prob),
    }
