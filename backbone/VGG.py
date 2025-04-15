import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model


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
        # Global Average Pooling (GAP)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        #self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        # Redefine Fully Connected layers for flexible input
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),  # 512 là số kênh đầu ra từ feature extractor VGG16
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        # Handle attention
        if isinstance(attention, nn.Module):
            self.attention_module = attention
            #self._replace_init_block_with_attention()
            self._insert_attention_after_block5()
        else:
            self.attention_module = None  # No attention by default

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
        x = self.vgg16.features(x)  # Pass through feature extractor
        x = self.global_avg_pool(x)  # GAP
        x = torch.flatten(x, 1)  # Flatten to shape (batch_size, 512)
        x = self.classifier(x)  # Pass through classifier
        return x


if __name__ == '__main__':
    # Import custom attention modules
    from attention.CBAM import CBAMBlock
    from attention.BAM import BAMBlock
    from attention.scSE import scSEBlock

    # Initialize models with different attention mechanisms
    cbam_module = CBAMBlock(channel=512, reduction=16, kernel_size=7)
    model_cbam = VGG16(pretrained=False, attention=cbam_module)

    bam_module = BAMBlock(channel=512, reduction=16, dia_val=2)
    model_bam = VGG16(pretrained=False, attention=bam_module)

    scse_module = scSEBlock(channel=512)
    model_scse = VGG16(pretrained=False, attention=scse_module)

    # Test model without attention
    model_no_attention = VGG16(pretrained=False)

    # Input tensor
    x = torch.randn(10, 3, 224, 224)

    # Test models
    outputs_cbam = model_cbam(x)
    print("CBAM Output Shape:", outputs_cbam.shape)

    outputs_bam = model_bam(x)
    print("BAM Output Shape:", outputs_bam.shape)

    outputs_scse = model_scse(x)
    print("scSE Output Shape:", outputs_scse.shape)

    outputs_no_attention = model_no_attention(x)
    print("No Attention Output Shape:", outputs_no_attention.shape)

    print(model_cbam)
