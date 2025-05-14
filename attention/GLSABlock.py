import torch
import torch.nn as nn
from .CBAM import CBAMBlock
from .BAM import BAMBlock
from .scSE import scSEBlock

class GLSABlock(nn.Module):
    """
    MHA → Stacked Attention block.
    - projects X→Q,K,V via 1×1 conv
    - performs multi-head self-attention
    - residual + LayerNorm
    - followed by one of {CBAM, BAM, scSE}
    """
    def __init__(self, channels, num_heads=8, attn_type='cbam', reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.C = channels
        self.to_qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.mha = nn.MultiheadAttention(embed_dim=channels,
                                         num_heads=num_heads,
                                         batch_first=True,
                                         bias=False)
        self.norm = nn.LayerNorm(channels)
        if attn_type == 'CBAM':
            self.attn2 = CBAMBlock(channel=channels, reduction=reduction_ratio, kernel_size=7)
        elif attn_type == 'BAM':
            self.attn2 = BAMBlock(channel=channels, reduction=reduction_ratio)
        elif attn_type == 'scSE':
            self.attn2 = scSEBlock(channel=channels, reduction_ratio=reduction_ratio)
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")
        self.bn_out = nn.BatchNorm2d(channels)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.hook_x = nn.Identity()
        self.hook_f1 = nn.Identity()
        self.hook_f2 = nn.Identity()
        self.hook_out = nn.Identity()
        self.hook_res = nn.Identity()

        # 1x1 conv để reprojection sau fusion
        self.to_out_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        # x = self.hook_x(x)
        # B, C, H, W = x.shape
        # qkv = self.to_qkv(x)
        # q, k, v = qkv.chunk(3, dim=1)
        # q = q.flatten(2).permute(0, 2, 1)
        # k = k.flatten(2).permute(0, 2, 1)
        # v = v.flatten(2).permute(0, 2, 1)
        # attn_out, _ = self.mha(q, k, v)
        # attn_out = self.norm(attn_out)
        # attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)
        # #attn_out = self.hook_f1(attn_out)
        # f1 = x + attn_out
        # f1 = self.hook_f1(f1)
        #
        # f2 = self.attn2(f1)
        # f2 = self.hook_f2(f2)
        #
        # # nhánh residual riêng
        # res = self.gamma * f2
        # res = self.hook_res(res)  # hook_res: chỉ γ·f2
        #
        # out = x + res
        # out = self.hook_out(out)  # hook_out
        #
        # Hook đầu vào
        x_in = self.hook_x(x)

        B, C, H, W = x_in.shape
        # 1) Multi-Head Self-Attention
        qkv = self.to_qkv(x_in)  # [B, 3C, H, W]
        q, k, v = qkv.chunk(3, dim=1)  # mỗi [B, C, H, W]
        # flatten và transpose để MultiheadAttention
        q = q.flatten(2).permute(0, 2, 1)  # [B, N, C]
        k = k.flatten(2).permute(0, 2, 1)
        v = v.flatten(2).permute(0, 2, 1)

        attn_out, _ = self.mha(q, k, v)  # [B, N, C]
        # residual + LayerNorm
        attn_out = self.norm(attn_out + q)  # vẫn [B, N, C]
        A = attn_out.permute(0, 2, 1).view(B, C, H, W)  # A′

        # 2) Local Attention
        f1 = x_in + A  # f1: sau MHA-residual
        f1 = self.hook_f1(f1)
        L = self.attn2(f1)  # L = f2
        L = self.hook_f2(L)

        # 3) Multiplicative Fusion
        fuse = A * L  # element-wise
        # reprojection
        R = self.to_out_conv(fuse)  # G
        R = self.hook_res(R)

        # 4) Final residual
        out = x_in + R
        out = self.hook_out(out)
        return out