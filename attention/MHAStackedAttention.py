import torch
import torch.nn as nn
from .CBAM import CBAMBlock
from .BAM import BAMBlock
from .scSE import scSEBlock

class MHAStackedAttention(nn.Module):
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

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.flatten(2).permute(0, 2, 1)
        k = k.flatten(2).permute(0, 2, 1)
        v = v.flatten(2).permute(0, 2, 1)
        attn_out, _ = self.mha(q, k, v)
        attn_out = self.norm(attn_out)
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)
        f1 = x + attn_out
        f2 = self.attn2(f1)
        out = self.bn_out(f2)
        return out