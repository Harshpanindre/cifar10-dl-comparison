"""Config loading, seeding, optimiser / scheduler builders and small helpers."""
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


# ---------------------------------------------------------------- config
def deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_overrides(cfg, overrides):
    """overrides: list like ["train.epochs=10", "optim.lr=0.01"]"""
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def load_config(model_name, config_path=None, overrides=None):
    with open(CONFIG_DIR / "common.yaml") as f:
        common = yaml.safe_load(f)
    path = Path(config_path) if config_path else CONFIG_DIR / f"{model_name}.yaml"
    with open(path) as f:
        model_cfg = yaml.safe_load(f)
    cfg = deep_merge(common, model_cfg)
    return apply_overrides(cfg, overrides)


# ---------------------------------------------------------------- misc
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------- optimisation
def build_optimizer(model, optim_cfg):
    """Weight decay is not applied to biases, norm layers, CLS token or position embeddings."""
    wd = float(optim_cfg.get("weight_decay", 0.0))
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "pos_embed" in name or "cls_token" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": wd},
              {"params": no_decay, "weight_decay": 0.0}]

    name = optim_cfg.get("name", "adam").lower()
    lr = float(optim_cfg["lr"])
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=float(optim_cfg.get("momentum", 0.9)),
                               nesterov=bool(optim_cfg.get("nesterov", False)))
    if name == "adam":
        return torch.optim.Adam(groups, lr=lr)
    if name == "adamw":
        betas = tuple(optim_cfg.get("betas", (0.9, 0.999)))
        return torch.optim.AdamW(groups, lr=lr, betas=betas)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, sched_cfg, steps_per_epoch, epochs):
    """Per-step LR schedule: optional linear warm-up followed by cosine decay."""
    sched_cfg = sched_cfg or {}
    if sched_cfg.get("name", "cosine") == "none":
        return None
    total = max(1, epochs * steps_per_epoch)
    warmup = int(sched_cfg.get("warmup_epochs", 0) * steps_per_epoch)
    min_ratio = float(sched_cfg.get("min_lr_ratio", 0.0))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
