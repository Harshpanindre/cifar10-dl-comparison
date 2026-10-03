"""Shared data pipeline (identical for all four models).

- CIFAR-10 is downloaded through torchvision.
- The 50k official training images are split 45k / 5k (stratified, fixed seed).
- The 10k official test images are only used by evaluate.py.
- Augmentation is applied to the training split only.
"""
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
NUM_CLASSES = len(CLASSES)

# Per-channel statistics of the CIFAR-10 training set
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class Cutout:
    """Zero out one random square of side `length` in a normalised CxHxW tensor."""

    def __init__(self, length=16):
        self.length = length

    def __call__(self, img):
        _, h, w = img.shape
        cy, cx = np.random.randint(h), np.random.randint(w)
        y1, y2 = max(0, cy - self.length // 2), min(h, cy + self.length // 2)
        x1, x2 = max(0, cx - self.length // 2), min(w, cx + self.length // 2)
        img = img.clone()
        img[:, y1:y2, x1:x2] = 0.0
        return img


def build_transforms(data_cfg):
    train_ops = []
    if data_cfg.get("augment", True):
        train_ops += [transforms.RandomCrop(32, padding=4),
                      transforms.RandomHorizontalFlip()]
    train_ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if data_cfg.get("cutout", False):
        train_ops.append(Cutout(data_cfg.get("cutout_length", 16)))

    eval_ops = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def _split_indices(n, val_size, seed, targets=None):
    idx = np.arange(n)
    tr_idx, va_idx = train_test_split(
        idx, test_size=val_size, random_state=seed,
        stratify=targets if targets is not None else None)
    return tr_idx.tolist(), va_idx.tolist()


def get_datasets(data_cfg, seed=42, fake=False, fake_size=2000):
    """Return (train_set, val_set, test_set).

    fake=True uses torchvision FakeData (random 32x32 images) so the whole
    pipeline can be tested without downloading CIFAR-10.
    """
    train_tf, eval_tf = build_transforms(data_cfg)

    if fake:
        kw = dict(size=fake_size, image_size=(3, 32, 32), num_classes=NUM_CLASSES)
        train_aug = datasets.FakeData(transform=train_tf, **kw)
        train_eval = datasets.FakeData(transform=eval_tf, **kw)
        test_set = datasets.FakeData(size=fake_size // 4, image_size=(3, 32, 32),
                                     num_classes=NUM_CLASSES, transform=eval_tf,
                                     random_offset=10_000)
        val_size = max(1, fake_size // 10)
        tr_idx, va_idx = _split_indices(fake_size, val_size, seed)
    else:
        root = data_cfg.get("root", "./data")
        # Two views of the same 50k images: augmented (train) and plain (val)
        train_aug = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
        train_eval = datasets.CIFAR10(root, train=True, download=True, transform=eval_tf)
        test_set = datasets.CIFAR10(root, train=False, download=True, transform=eval_tf)
        tr_idx, va_idx = _split_indices(len(train_aug), data_cfg.get("val_size", 5000),
                                        seed, targets=train_aug.targets)

    return Subset(train_aug, tr_idx), Subset(train_eval, va_idx), test_set


def get_loaders(data_cfg, seed=42, fake=False):
    train_set, val_set, test_set = get_datasets(data_cfg, seed, fake)
    bs = data_cfg.get("batch_size", 128)
    nw = data_cfg.get("num_workers", 2)
    pin = torch.cuda.is_available()
    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, generator=g,
                              num_workers=nw, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False,
                             num_workers=nw, pin_memory=pin)
    return train_loader, val_loader, test_loader


def get_raw_test_images(data_cfg, fake=False, fake_size=2000):
    """Un-normalised test images (uint8, N x 32 x 32 x 3) in the same order as the test loader."""
    if fake:
        ds = datasets.FakeData(size=fake_size // 4, image_size=(3, 32, 32),
                               num_classes=NUM_CLASSES, random_offset=10_000)
        return np.stack([np.array(ds[i][0]) for i in range(len(ds))])
    ds = datasets.CIFAR10(data_cfg.get("root", "./data"), train=False, download=True)
    return ds.data
