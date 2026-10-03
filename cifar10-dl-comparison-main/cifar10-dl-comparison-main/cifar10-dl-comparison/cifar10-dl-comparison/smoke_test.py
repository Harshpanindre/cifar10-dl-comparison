"""Fast sanity check — no dataset download, runs on CPU in seconds.

    python smoke_test.py

For every model: build from its config, run a forward + backward pass on random
images, check the output shape, and print the parameter count. Then run one
tiny training step through the real engine on random data.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.engine import evaluate, train_one_epoch
from src.metrics import compute_metrics
from src.data import CLASSES
from src.models import DISPLAY_NAMES, MODEL_REGISTRY, build_model
from src.models.lstm import image_to_sequence
from src.utils import build_optimizer, build_scheduler, count_parameters, load_config


def check_sequence_shapes():
    x = torch.randn(2, 3, 32, 32)
    assert image_to_sequence(x, "rows").shape == (2, 32, 96)
    assert image_to_sequence(x, "patches", 4).shape == (2, 64, 48)
    # first patch must equal the top-left 4x4 block (channel-major)
    first = image_to_sequence(x, "patches", 4)[0, 0]
    assert torch.allclose(first, x[0, :, :4, :4].reshape(-1))
    print("sequence reshaping OK")


def main():
    torch.manual_seed(0)
    check_sequence_shapes()
    device = torch.device("cpu")
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(torch.randn(64, 3, 32, 32), torch.randint(0, 10, (64,))),
                        batch_size=16, drop_last=True)

    for name in MODEL_REGISTRY:
        cfg = load_config(name)
        model = build_model(cfg["model"]).to(device)
        out = model(x)
        assert out.shape == (8, 10), f"{name}: bad output shape {tuple(out.shape)}"
        nn.CrossEntropyLoss()(out, y).backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad), f"{name}: missing grads"

        opt = build_optimizer(model, cfg["optim"])
        sched = build_scheduler(opt, cfg.get("scheduler"), len(loader), 1)
        tr = train_one_epoch(model, loader, nn.CrossEntropyLoss(), opt, device, None, sched,
                             grad_clip=float(cfg["train"].get("grad_clip", 0)))
        ev = evaluate(model, loader, nn.CrossEntropyLoss(), device)
        compute_metrics(ev["y_true"], ev["y_pred"], CLASSES)
        print(f"{DISPLAY_NAMES[name]:<20} params {count_parameters(model) / 1e6:6.2f}M | "
              f"output {tuple(out.shape)} | train loss {tr['loss']:.3f}  OK")
    print("\nAll models passed the smoke test.")


if __name__ == "__main__":
    main()
