"""Model 2 — ResNet-style CNN (convolutional family).
Owner: Member 2 — Sanket Tippimani

How it uses the image:
    The image stays a 2-D grid. Small 3x3 filters slide over it, so the same
    weights detect an edge or texture wherever it appears (weight sharing ->
    translation equivariance). Stride-2 stages shrink the grid 32 -> 16 -> 8
    while increasing channels, building edges -> parts -> objects. Residual
    shortcuts (He et al., 2016) let gradients flow through deeper stacks.
"""
import torch.nn as nn


class BasicBlock(nn.Module):
    """conv3x3-BN-ReLU-conv3x3-BN + shortcut, then ReLU."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            # 1x1 projection so the shapes match for the addition
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNetCIFAR(nn.Module):
    def __init__(self, num_classes=10, channels=(64, 128, 256), blocks_per_stage=(2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True))

        stages, in_ch = [], channels[0]
        for i, (ch, n) in enumerate(zip(channels, blocks_per_stage)):
            stride = 1 if i == 0 else 2           # 32x32 -> 16x16 -> 8x8
            blocks = [BasicBlock(in_ch, ch, stride)]
            blocks += [BasicBlock(ch, ch, 1) for _ in range(n - 1)]
            stages.append(nn.Sequential(*blocks))
            in_ch = ch
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(in_ch, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):                 # (B, 3, 32, 32)
        x = self.stem(x)                  # (B, 64, 32, 32)
        x = self.stages(x)                # (B, 256, 8, 8)
        x = self.pool(x).flatten(1)       # (B, 256)
        return self.head(x)
