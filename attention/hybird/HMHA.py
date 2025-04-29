import torch
import torch.nn as nn
import torch.nn.functional as F

class HMHA(nn.Module):
    def __init__(self, channel, num_heads=8, reduction=16, pool_size=(16, 16)):
        super(HMHA, self).__init__()
        self.num_heads = num_heads
        self.pool_size = pool_size
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

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

        # === 1. Reduce spatial size before attention ===
        x_reduced = F.adaptive_avg_pool2d(x, output_size=self.pool_size)
        _, _, h_p, w_p = x_reduced.shape

        # === 2. QKV Projection ===
        qkv = self.qkv(x_reduced)
        q, k, v = qkv.chunk(3, dim=1)

        # === 3. Reshape for multi-head attention ===
        q = q.view(b, self.num_heads, self.dim_per_head, h_p * w_p).transpose(-2, -1)  # B, heads, HW, dim
        k = k.view(b, self.num_heads, self.dim_per_head, h_p * w_p).transpose(-2, -1)
        v = v.view(b, self.num_heads, self.dim_per_head, h_p * w_p).transpose(-2, -1)

        # === 4. Attention calculation ===
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-1, -2)) * self.temperature  # B, heads, HW, HW
        attn = F.softmax(attn, dim=-1)

        out = attn @ v  # B, heads, HW, dim
        out = out.transpose(-2, -1).reshape(b, self.total_dim, h_p, w_p)

        # === 5. Upsample back to original size ===
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)

        # === 6. Final projection ===
        out = self.project_out(out)
        return out
