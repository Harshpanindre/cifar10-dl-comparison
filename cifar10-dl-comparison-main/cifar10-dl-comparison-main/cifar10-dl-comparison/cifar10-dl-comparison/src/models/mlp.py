"""Model 1 — MLP baseline (fully connected family).
Owner: Member 1 — Harsh Kushwaha

How it uses the image:
    The 3x32x32 image is flattened into one 3,072-value vector. Every pixel is
    connected to every hidden unit, so the model has NO notion of which pixels
    are neighbours: shuffling the pixel order (the same way for every image)
    would not change what it can learn. This is the "no spatial inductive bias"
    baseline the other three models are compared against.
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, num_classes=10, input_dim=3 * 32 * 32,
                 hidden_dims=(1024, 512, 256), dropout=0.3):
        super().__init__()
        layers, d = [], input_dim
        for h in hidden_dims:
            # Linear -> BatchNorm -> ReLU -> Dropout
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = h
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(d, num_classes)

    def forward(self, x):                 # x: (B, 3, 32, 32)
        x = torch.flatten(x, 1)           # (B, 3072)
        return self.head(self.features(x))
