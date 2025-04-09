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
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Redefine Fully Connected layers for flexible input
        self.vgg16.output = nn.Sequential(
            nn.Linear(512, 2048),  # 512 là số kênh đầu ra từ feature extractor VGG16
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, num_classes),
        )
        # Handle attention
        if isinstance(attention, nn.Module):
            self.attention_module = attention
            self._replace_init_block_with_attention()
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

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The model output.
        """
        x = self.vgg16.features(x)  # Pass through feature extractor
        x = self.global_pool(x)  # Apply global average pooling
        x = torch.flatten(x, 1)  # Flatten to shape (batch_size, 512)
        x = self.vgg16.output(x)  # Pass through classifier
        return x


if __name__ == '__main__':
    # Import custom attention modules
    from attention.CBAM import CBAMBlock
    from attention.BAM import BAMBlock
    from attention.scSE import scSEBlock

    # Initialize models with different attention mechanisms
    cbam_module = CBAMBlock(channel=3, reduction=16, kernel_size=7)
    model_cbam = VGG16(pretrained=False, attention=cbam_module)

    bam_module = BAMBlock(channel=3, reduction=16, dia_val=2)
    model_bam = VGG16(pretrained=False, attention=bam_module)

    scse_module = scSEBlock(channel=3)
    model_scse = VGG16(pretrained=False, attention=scse_module)

    # Test model without attention
    model_no_attention = VGG16(pretrained=False)

    # Input tensor
    x = torch.randn(10, 3, 128, 128)

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