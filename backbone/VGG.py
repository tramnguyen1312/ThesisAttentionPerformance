import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
from attention.hybird.HMHA_CBAM import HMHA_CBAM
import torch.nn.functional as F
from torchvision.models import resnet50, vgg16
from attention.MHAStackedAttention import MHAStackedAttention

class VGG16(torch.nn.Module):
    def __init__(self, attn_type='cbam', num_heads=8, weights=None, num_classes=1000):
        super().__init__()
        backbone = vgg16(weights=weights)
        feats = backbone.features
        # up to pool4
        self.features1 = nn.Sequential(*feats[:24])
        C4 = 512
        # self.mha_block = MHAStackedAttention(
        #     channels=C4,
        #     num_heads=num_heads,
        #     attn_type=attn_type
        # )
        # # conv5_x + pool5
        # self.features2 = nn.Sequential(*feats[24:])
        # self.classifier = backbone.classifier

        # remaining conv5_x and pool5
        self.features2 = nn.Sequential(*feats[24:])
        # classifier: replace last FC layer
        cls = list(backbone.classifier.children())
        in_features = cls[-1].in_features
        cls[0] = nn.Linear(512 * 7 * 7, 4096)
        cls[-1] = nn.Linear(in_features, num_classes)
        self.classifier = nn.Sequential(*cls)

    def forward(self, x):
        x = self.features1(x)
        # x = self.mha_block(x)
        x = self.features2(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)



if __name__ == '__main__':
    model_v = VGG16(attn_type='cbam', num_heads=8, weights=None, num_classes=10)
    inp = torch.randn(128, 3, 9, 224)
    out_r = model_v(inp)
    print(model_v)  # e.g. [2, 1000] each
    print(out_r)
