"""Model 3 — RNN/LSTM (recurrent family).
Owner: Member 3 — Sanyam Satija

How it uses the image:
    The image is turned into a SEQUENCE and read one step at a time.
      sequence="rows":    32 time steps, each step = one row of 32 pixels x 3 channels = 96 values
      sequence="patches": 64 time steps, each step = one 4x4 patch x 3 channels = 48 values
    The LSTM carries information forward in its hidden/cell state, so a pixel
    in row 30 only "meets" a pixel in row 2 through that memory. Images have
    no natural reading order, which is why this model is expected to be a
    structurally mismatched (but instructive) baseline.
"""
import torch
import torch.nn as nn


def image_to_sequence(x, mode="rows", patch_size=4):
    """(B, C, H, W) -> (B, T, F)"""
    b, c, h, w = x.shape
    if mode == "rows":
        # each row becomes one time step: (B, H, W*C)
        return x.permute(0, 2, 3, 1).reshape(b, h, w * c)
    if mode == "patches":
        p = patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)          # (B, C, H/p, W/p, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5)                 # (B, H/p, W/p, C, p, p)
        return x.reshape(b, (h // p) * (w // p), c * p * p)  # raster order of patches
    raise ValueError(f"Unknown sequence mode: {mode}")


class SequenceLSTM(nn.Module):
    def __init__(self, num_classes=10, sequence="rows", patch_size=4, image_size=32,
                 in_channels=3, hidden_size=256, num_layers=2, bidirectional=False, dropout=0.2):
        super().__init__()
        self.sequence = sequence
        self.patch_size = patch_size
        input_size = image_size * in_channels if sequence == "rows" else in_channels * patch_size ** 2

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=bidirectional)
        self.num_dirs = 2 if bidirectional else 1
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(hidden_size * self.num_dirs, num_classes))
        self._init_forget_bias()

    def _init_forget_bias(self):
        # Gate order in PyTorch is (input, forget, cell, output).
        # A forget-gate bias of 1 helps the LSTM keep memory early in training.
        for name, param in self.lstm.named_parameters():
            if name.startswith("bias_ih"):
                h = param.shape[0] // 4
                with torch.no_grad():
                    param[h:2 * h].fill_(1.0)

    def forward(self, x):                                   # (B, 3, 32, 32)
        seq = image_to_sequence(x, self.sequence, self.patch_size)   # (B, T, F)
        _, (h_n, _) = self.lstm(seq)                        # h_n: (layers*dirs, B, H)
        if self.num_dirs == 2:
            feat = torch.cat([h_n[-2], h_n[-1]], dim=1)     # last layer, both directions
        else:
            feat = h_n[-1]                                  # last layer, final step
        return self.head(feat)
