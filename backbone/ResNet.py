import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from torchvision.models import resnet50, vgg16
from attention import GLSABlock

class ResNet18(nn.Module):
    def __init__(self, attn_type='CBAM', num_heads=8, pretrained=True, num_classes=1000):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3
        )
        C3 = backbone.layer3[-1].conv2.out_channels
        self.attn_type = attn_type

        if self.attn_type != 'none':
            self.mha_block = GLSABlock(
                channels=C3,
                num_heads=num_heads,
                attn_type=attn_type
            )
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = self.stem(x)
        if self.attn_type != 'none':
            x = self.mha_block(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

if __name__ == '__main__':
    model_v = ResNet18(attn_type='CBAM', num_heads=8, pretrained=False, num_classes=10)
    inp = torch.randn(128, 3, 224, 224)
    out_r = model_v(inp)
    print(model_v)
    print(out_r)