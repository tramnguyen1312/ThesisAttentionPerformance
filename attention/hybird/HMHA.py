import torch
import torch.nn as nn
import torch.nn.functional as F

class HMHA(nn.Module):
    def __init__(self, channel, num_heads=8, reduction=16):
        super(HMHA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Giảm chiều để tiết kiệm tính toán
        self.dim_per_head = max(channel // reduction // num_heads, 1)
        self.total_dim = self.dim_per_head * num_heads

        # Query, Key, Value projections cho tất cả heads
        self.qkv = nn.Conv2d(channel, 3 * self.total_dim, kernel_size=1, bias=False)

        # Projection sau khi concat tất cả heads
        self.project_out = nn.Sequential(
            nn.Conv2d(self.total_dim, channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        # QKV projection
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape để tính multi-head attention
        q = q.view(b, self.num_heads, self.dim_per_head, h, w)
        k = k.view(b, self.num_heads, self.dim_per_head, h, w)
        v = v.view(b, self.num_heads, self.dim_per_head, h, w)

        # Flatten spatial dimensions
        q = q.flatten(3)  # B, num_heads, dim_per_head, H*W
        k = k.flatten(3)
        v = v.flatten(3)

        # Transpose for attention calculation
        q = q.transpose(-2, -1)  # B, num_heads, H*W, dim_per_head
        k = k.transpose(-2, -1)  # B, num_heads, H*W, dim_per_head
        v = v.transpose(-2, -1)  # B, num_heads, H*W, dim_per_head

        # Attention map calculation
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # B, num_heads, H*W, H*W
        attn = F.softmax(attn, dim=-1)

        # Apply attention to value
        out = (attn @ v)  # B, num_heads, H*W, dim_per_head

        # Reshape back to spatial dimensions
        #out = out.transpose(-2, -1).view(b, self.total_dim, h, w)
        out = out.transpose(-2, -1).reshape(b, self.total_dim, h, w)
        # Project back to channel dimension
        out = self.project_out(out)

        return out
