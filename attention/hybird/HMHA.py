import torch
import torch.nn as nn
import torch.nn.functional as F


class HMHA(nn.Module):
    def __init__(self, channel, num_heads=8, reduction=16, use_residual=True):
        super(HMHA, self).__init__()
        self.num_heads = num_heads
        self.use_residual = use_residual

        self.dim_per_head = max(channel // reduction // num_heads, 1)
        self.total_dim = self.dim_per_head * num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # Learnable scale

        # Linear projections
        self.qkv_conv = nn.Conv2d(channel, 3 * self.total_dim, kernel_size=1, bias=False)

        # Output projection
        self.project_out = nn.Sequential(
            nn.Conv2d(self.total_dim, channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.Dropout2d(0.1)
        )

        # Optional residual block
        if self.use_residual:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_conv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head: (B, H, D, H, W)
        q = q.view(b, self.num_heads, self.dim_per_head, h * w).transpose(-2, -1)  # (B, H, HW, D)
        k = k.view(b, self.num_heads, self.dim_per_head, h * w).transpose(-2, -1)
        v = v.view(b, self.num_heads, self.dim_per_head, h * w).transpose(-2, -1)

        # Attention computation
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # (B, H, HW, HW)
        attn = F.softmax(attn, dim=-1)

        out = attn @ v  # (B, H, HW, D)
        out = out.transpose(-2, -1).reshape(b, self.total_dim, h, w)  # Merge heads

        out = self.project_out(out)

        if self.use_residual:
            out = out + self.residual(x)

        return out
