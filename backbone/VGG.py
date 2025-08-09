import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
# from attention.hybird.HMHA_CBAM import HMHA_CBAM
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
from attention import GLSABlock, MHAOnlyBlock, LightweightOnlyBlock

class VGG16(torch.nn.Module):
    def __init__(self, attn_type='CBAM', num_heads=8, pretrained=None, num_classes=1000, fusion_type='multiply'):
        super().__init__()
        backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        feats = backbone.features
        # up to pool4
        self.features1 = nn.Sequential(*feats[:24])
        C4 = 512
        self.attn_type = attn_type

        if self.attn_type != 'none':
            if attn_type == 'mha_only':
                self.mha_block = MHAOnlyBlock(channels=C4, num_heads=num_heads)
            elif attn_type in ['cbam_only', 'bam_only', 'scse_only']:
                lightweight_type = attn_type.replace('_only', '').upper()
                self.mha_block = LightweightOnlyBlock(channels=C4, attn_type=lightweight_type)
            else:  # Full GLA-Block
                self.mha_block = GLSABlock(
                    channels=C4,
                    num_heads=num_heads,
                    attn_type=attn_type,
                    fusion_type=fusion_type
                )
        self.features2 = nn.Sequential(*feats[24:])
        cls = list(backbone.classifier.children())
        in_features = cls[-1].in_features
        cls[-1] = nn.Linear(in_features, num_classes)
        self.classifier = nn.Sequential(*cls)

        self._print_num_params()

    def _print_num_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[VGG16] Total params: {total:,} | Trainable: {trainable:,}")

    def forward(self, x):
        x = self.features1(x)
        if self.attn_type != 'none':
            x = self.mha_block(x)
        x = self.features2(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)



if __name__ == '__main__':
    model_v = VGG16(attn_type='scSE',  pretrained=True, num_heads=8, num_classes=10)
    inp = torch.randn(64, 3, 224, 224)
    out_r = model_v(inp)
    print(model_v)  # e.g. [2, 1000] each
    # print(out_r)