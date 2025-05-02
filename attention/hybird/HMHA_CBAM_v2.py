import torch
import torch.nn as nn
import torch.nn.functional as F


class HMHA_CBAM_v2(nn.Module):
    def __init__(self, channel, num_heads=8, reduction=16, kernel_size=7, attention_type='CBAM'):
        super(HMHA_CBAM_v2, self).__init__()

        # Chia feature ra làm 2 nhánh
        self.split_channels = channel // 2
        self.remain_channels = channel - self.split_channels

        # HMHA áp dụng lên 1 nửa
        from attention.hybird.HMHA import HMHA
        self.mha = HMHA(self.split_channels, num_heads=num_heads, reduction=reduction)

        # Chọn attention module (CBAM, BAM, scSE)
        if attention_type.upper() == 'CBAM':
            from attention.CBAM import CBAMBlock
            self.second_attn = CBAMBlock(channel=self.remain_channels, reduction=16, kernel_size=7)
        elif attention_type.upper() == 'BAM':
            from attention.BAM import BAMBlock
            self.second_attn = BAMBlock(channel=self.remain_channels, reduction=16, dia_val=2)
        elif attention_type.upper() == 'SCSE':
            from attention.scSE import scSEBlock
            self.second_attn = scSEBlock(channel=self.remain_channels, reduction_ratio=2)
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")

        # Fusion weights học được (scalar)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        self.fusion_bn = nn.BatchNorm2d(channel)
        self.fusion_act = nn.ReLU(inplace=True)

    def forward(self, x):
        # Split input
        x1, x2 = torch.split(x, [self.split_channels, self.remain_channels], dim=1)

        # Apply attention riêng biệt
        out1 = self.mha(x1)
        out2 = self.second_attn(x2)

        # Concatenate lại
        out = torch.cat([self.alpha * out1, self.beta * out2], dim=1)

        # Normalize và kích hoạt
        out = self.fusion_bn(out)
        out = self.fusion_act(out)
        return out
