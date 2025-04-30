import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
from attention.hybird.HMHA_CBAM import HMHA_CBAM


class VGG16(torch.nn.Module):
    def __init__(self, pretrained=False, attention=None, num_classes=10):
        """
        VGG16 model with optional attention mechanism at the input.

        Args:
            pretrained (bool): Whether to load pretrained weights for VGG16.
            attention (nn.Module, optional): An instantiated attention module to add to the input.
                                              If None, no attention module is used.
        """
        super().__init__()
        # Load the base VGG16 model
        self.vgg16 = ptcv_get_model("vgg16", pretrained=pretrained)
        self.attention_module = attention
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512 + 512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def _replace_init_block_with_attention(self):
        """
        Replace the initial layer in VGG16 by adding attention before the first convolution block.
        """
        original_conv_block = self.vgg16.features[0]  # First convolution layer
        self.vgg16.features[0] = nn.Sequential(
            self.attention_module,  # Apply the attention module first
            original_conv_block  # Follow with the original convolution
        )

    def _insert_attention_after_block4(self):
        """
        Insert the attention mechanism after Block 4 and before Block 5.
        """
        # Split the features into Block 4, Attention, and Block 5+
        block4 = self.vgg16.features[:24]  # Block 4 ends before index 24 (after 3rd conv in Block 4)
        block5 = self.vgg16.features[24:]  # Block 5 starts from index 24

        # Combine Block 4, Attention, Block 5
        self.vgg16.features = nn.Sequential(
            *block4,  # All layers in Block 4
            self.attention_module,  # Insert attention module here
            *block5  # All layers in Block 5 and beyond
        )

    def _insert_attention_after_block5(self):
        """
        Insert the attention mechanism after Block 5.
        """
        # Block 5 index: VGG16 standard features end at index 30 after 5 blocks
        block5 = self.vgg16.features[:30]  # The first 30 layers correspond to VGG's Block 5 (5 conv layers, pool)
        rest = self.vgg16.features[30:]  # All layers after Block 5

        # Combine Block 5, Attention, and remaining layers
        self.vgg16.features = nn.Sequential(
            *block5,  # All layers before adding Attention
            self.attention_module,  # Insert attention module here
            *rest  # All layers after Block 5
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The model output.
        """
        # x = self.vgg16.features(x)  # Pass through feature extractor
        # x = self.adaptive_avg_pool(x)  # AAP
        # x = torch.flatten(x, 1)  # Flatten to shape (batch_size, 512)
        # x = self.classifier(x)  # Pass through classifier
        # Stage 1 (thường conv + relu + pool)
        x = self.vgg16.features.stage1(x)  # (batch, 64, H/2, W/2)
        # Stage 2
        x = self.vgg16.features.stage2(x)  # (batch, 128, H/4, W/4)

        x = self.vgg16.features.stage3(x)

        x = self.vgg16.features.stage4(x)

        # Clone để tránh xung đột gradient
        x_for_att = x.clone()

        # Nhánh giữa
        mid_feat = self.attention_module(x_for_att)  # (batch, 512, H/16, W/16)
        mid_feat = self.adaptive_avg_pool(mid_feat)
        mid_feat = mid_feat.view(mid_feat.size(0), -1)  # (batch, 512)

        # Nhánh cao
        high_feat = self.vgg16.features.stage5(x)
        high_feat = self.adaptive_avg_pool(high_feat)
        high_feat = high_feat.view(high_feat.size(0), -1)

        # Kết hợp
        fused_feat = torch.cat([mid_feat, high_feat], dim=1)
        x = self.classifier(fused_feat)

        return x


if __name__ == '__main__':
    # Import custom attention modules
    mha_cbam = HMHA_CBAM(channel=512, num_heads=8, reduction=16, kernel_size=7)
    model = VGG16(pretrained=False, attention=mha_cbam, num_classes=10)

    x = torch.randn(8, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)
