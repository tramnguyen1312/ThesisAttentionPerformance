import torch
import torch.nn as nn
from attention.CBAM import CBAMBlock
from attention.hybird.HMHA import HMHA

class HMHA_CBAM(nn.Module):
    def __init__(self, channel, num_heads=8, reduction=16, kernel_size=7):
        super(HMHA_CBAM, self).__init__()

        print(f'channel: {channel}')

        # Custom MHA
        self.mha = HMHA(channel, num_heads, reduction)

        # CBAM
        self.cbam = CBAMBlock(channel, reduction, kernel_size)

        # Spatial attention cho fusion
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # Channel gate cho fusion
        self.channel_gate = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # MHA path
        mha_out = self.mha(x)

        # CBAM path
        cbam_out = self.cbam(x)

        # Spatial attention map cho fusion
        avg_pool = torch.mean(torch.cat([mha_out, cbam_out], dim=1), dim=1, keepdim=True)
        max_pool, _ = torch.max(torch.cat([mha_out, cbam_out], dim=1), dim=1, keepdim=True)
        spatial_map = self.spatial_attn(torch.cat([avg_pool, max_pool], dim=1))

        # Channel attention map cho fusion
        b, c, _, _ = x.shape
        channel_info = torch.cat([
            torch.mean(mha_out, dim=[2, 3]),
            torch.mean(cbam_out, dim=[2, 3])
        ], dim=1).reshape(b, 1, c * 2)
        channel_weight = self.channel_gate(channel_info).reshape(b, c * 2, 1, 1)
        mha_weight, cbam_weight = channel_weight.chunk(2, dim=1)

        # Adaptive fusion
        output = mha_out * mha_weight * spatial_map + cbam_out * cbam_weight * spatial_map

        return output
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from attention.CBAM import CBAMBlock
# from attention.hybird.HMHA import HMHA
#
# class HMHA_CBAM(nn.Module):
#     def __init__(self, channel, num_heads=8, reduction=16, kernel_size=7):
#         super(HMHA_CBAM, self).__init__()
#
#         self.mha = HMHA(channel, num_heads, reduction)
#         self.cbam = CBAMBlock(channel, reduction, kernel_size)
#
#         # Fusion weights (learnable scalar)
#         self.alpha = nn.Parameter(torch.tensor(0.5))  # weight for MHA
#         self.beta = nn.Parameter(torch.tensor(0.5))   # weight for CBAM
#
#         # Optional normalization + activation
#         self.fusion_bn = nn.BatchNorm2d(channel)
#         self.fusion_act = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         mha_out = self.mha(x)
#         cbam_out = self.cbam(x)
#
#         # Normalize each path before fusion (optional but helpful)
#         mha_out = F.normalize(mha_out, dim=1)
#         cbam_out = F.normalize(cbam_out, dim=1)
#
#
#         # Learnable weighted fusion
#         fused = self.alpha * mha_out + self.beta * cbam_out
#
#         # Optional final processing
#         fused = self.fusion_bn(fused)
#         fused = self.fusion_act(fused)
#
#         return fused