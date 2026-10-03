"""Model registry: build any of the four models from its config block."""
from .cnn import ResNetCIFAR
from .lstm import SequenceLSTM
from .mlp import MLP
from .vit import ViT

MODEL_REGISTRY = {
    "mlp": MLP,            # Member 1
    "cnn": ResNetCIFAR,    # Member 2
    "lstm": SequenceLSTM,  # Member 3
    "vit": ViT,            # Member 4
}

DISPLAY_NAMES = {
    "mlp": "MLP baseline",
    "cnn": "CNN (ResNet-style)",
    "lstm": "RNN/LSTM",
    "vit": "Vision Transformer",
}


def build_model(model_cfg, num_classes=10):
    cfg = dict(model_cfg)
    name = cfg.pop("name")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")
    kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items()}
    return MODEL_REGISTRY[name](num_classes=num_classes, **kwargs)
