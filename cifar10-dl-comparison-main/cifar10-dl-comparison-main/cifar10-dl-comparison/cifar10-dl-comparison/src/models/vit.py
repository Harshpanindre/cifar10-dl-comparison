"""Model 4 — Vision Transformer (attention family).
Owner: Member 4 — Devansh Sharma

How it uses the image:
    The image is cut into non-overlapping 4x4 patches (8x8 = 64 patches). Each
    patch is linearly projected to a token, a learnable [CLS] token is added,
    and position embeddings tell the model where each patch came from.
    Multi-head self-attention then lets EVERY patch look at EVERY other patch
    in a single layer (global context from the first layer), unlike the CNN,
    which only sees a small neighbourhood per layer. The price is a weak
    built-in prior, so ViTs need more data / regularisation (Dosovitskiy et al., 2021).
"""
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Split into patches and project each to embed_dim (a strided conv does both at once)."""

    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):                          # (B, 3, 32, 32)
        x = self.proj(x)                           # (B, D, 8, 8)
        return x.flatten(2).transpose(1, 2)        # (B, 64, D)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x):                          # (B, N, D)
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]           # each (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, N, N)
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    """Pre-norm block: x + MHSA(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, dim, num_heads, mlp_ratio=2.0, dropout=0.1, attn_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, attn_dropout, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self, num_classes=10, image_size=32, patch_size=4, in_channels=3,
                 embed_dim=192, depth=6, num_heads=3, mlp_ratio=2.0,
                 dropout=0.1, attn_dropout=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):                                  # (B, 3, 32, 32)
        x = self.patch_embed(x)                            # (B, 64, D)
        cls = self.cls_token.expand(x.shape[0], -1, -1)    # (B, 1, D)
        x = torch.cat([cls, x], dim=1) + self.pos_embed    # (B, 65, D)
        x = self.blocks(self.pos_drop(x))
        x = self.norm(x)
        return self.head(x[:, 0])                          # classify from [CLS]
