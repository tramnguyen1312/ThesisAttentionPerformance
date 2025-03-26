import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
class Flatten(nn.Module):
    def forward(self,x):
        return x.view(x.shape[0],-1)

class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16,num_layers=3):
        reduced_channels = max(channel // reduction, 1)
        super().__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        gate_channels=[channel]
        gate_channels+=[reduced_channels]*num_layers
        gate_channels+=[channel]


        self.ca=nn.Sequential()
        self.ca.add_module('flatten',Flatten())
        for i in range(len(gate_channels)-2):
            self.ca.add_module('fc%d'%i,nn.Linear(gate_channels[i],gate_channels[i+1]))
            self.ca.add_module('bn%d'%i,nn.BatchNorm1d(gate_channels[i+1]))
            self.ca.add_module('relu%d'%i,nn.ReLU())
        self.ca.add_module('last_fc',nn.Linear(gate_channels[-2],gate_channels[-1]))
        

    def forward(self, x) :
        # res=self.avgpool(x)
        # res=self.ca(res)
        # res=res.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        # return res

        res = self.avgpool(x)  # [batch_size, channels, 1, 1]
        # Flatten and apply fully connected layers
        res = self.ca(res.view(res.size(0), -1))  # Flatten to [batch_size, channels]
        # Reshape back to [batch_size, channels, 1, 1] and expand to match x
        res = res.unsqueeze(-1).unsqueeze(-1).expand_as(x)  # Expand to [batch_size, channels, height, width]
        return res

class SpatialAttention(nn.Module):
    def __init__(self,channel,reduction=16,num_layers=3,dia_val=2):
        reduced_channels = max(channel // reduction, 1)
        super().__init__()
        self.sa=nn.Sequential()
        self.sa.add_module('conv_reduce1',nn.Conv2d(kernel_size=1,in_channels=channel,out_channels=reduced_channels))
        self.sa.add_module('bn_reduce1',nn.BatchNorm2d(reduced_channels))
        self.sa.add_module('relu_reduce1',nn.ReLU())
        for i in range(num_layers):
            self.sa.add_module('conv_%d'%i,nn.Conv2d(kernel_size=3,in_channels=reduced_channels,out_channels=reduced_channels,padding=1,dilation=dia_val))
            self.sa.add_module('bn_%d'%i,nn.BatchNorm2d(reduced_channels))
            self.sa.add_module('relu_%d'%i,nn.ReLU())
        self.sa.add_module('last_conv',nn.Conv2d(reduced_channels,1,kernel_size=1))

    def forward(self, x) :
        res = self.sa(x)  # [batch_size, 1, height', width']
        # Ensure output has same spatial size as input
        res = F.interpolate(res, size=x.size()[2:], mode='bilinear', align_corners=False)  # Resize to [batch_size, 1, height, width]
        res = res.expand_as(x)  # Match x's dimensions [batch_size, channels, height, width]
        return res




class BAMBlock(nn.Module):

    def __init__(self, channel=512,reduction=16,dia_val=2):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(channel=channel,reduction=reduction,dia_val=dia_val)
        self.sigmoid=nn.Sigmoid()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        sa_out=self.sa(x)
        ca_out=self.ca(x)
        weight=self.sigmoid(sa_out+ca_out)
        out=(1+weight)*x
        return out

if __name__ == '__main__':
    input=torch.randn(2,3,224,224)
    bam = BAMBlock(channel=3,reduction=16,dia_val=2)
    output=bam(input)
    print(output)

    