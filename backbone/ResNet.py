from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
import torch.nn as nn


class ResNet18(torch.nn.Module):
    def __init__(self, pretrained=False, attention=None, num_classes=10):
        """
        ResNet50 model with optional attention mechanism at the initial convolution block.

        Args:
            pretrained (bool): Whether to load pretrained weights for ResNet50.
            attention (nn.Module, optional): An attention module to add to the
                initial block. If None, no attention module is applied.
        """
        super().__init__()
        self.resnet = ptcv_get_model("resnet18", pretrained=pretrained)  # Load ResNet50 backbone
        # Remove Max-Pooling from the initial stage
        #self.resnet.features.init_block.pool = nn.Identity()  # Remove MaxPool2d
        # #
        # # Modify stride in res2 and res3 stages to retain more spatial information
        # self.resnet.features.init_block.conv = nn.Conv2d(
        #     in_channels=3,
        #     out_channels=64,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        #     bias=False,
        # )

        # Check if attention module is provided
        if isinstance(attention, nn.Module):
            self.attention_module = attention
            #self._replace_init_block_with_attention()
            self._insert_attention_after_block4()
        else:
            self.attention_module = None  # No attention by default

        # self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # GAP and Max Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        #self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten tensor
            nn.Linear(512, 512),  # ResNet18 outputs 512 channels
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),  # Optional dropout
            nn.Linear(512, num_classes)  # Final classification layer
        )

    def _replace_init_block_with_attention(self):
        """
        Replace the initial convolution block in ResNet50 with
        a sequential module that applies the attention module first.
        """
        original_conv_block = self.resnet.features.init_block.conv  # Original convolution
        self.resnet.features.init_block.conv = nn.Sequential(
            self.attention_module,  # Insert the attention module
            original_conv_block  # Follow with the original convolution
        )

    def _insert_attention_after_block4(self):
        """
        Inserts the attention mechanism after block 4 and before block 5.
        """
        # Split blocks into before, attention, and after
        res4 = self.resnet.features[3]  # res4
        res5 = self.resnet.features[4]  # res5
        # Replace block list with attention sandwiched between
        self.resnet.features[3] = nn.Sequential(
            res4,  # Original block 4
            self.attention_module  # Insert attention here
        )
        self.resnet.features[4] = res5  # Retain block 5

    def _insert_attention_after_block5(self):
        """
        Inserts the attention mechanism after block 5 and before the final pooling.
        """
        # Access the fifth block and the original output
        res5 = self.resnet.features[4]  # Get block 5
        # Assemble a Sequential container that includes block 5 and the attention module
        self.resnet.features[4] = nn.Sequential(
            res5,  # The original block 5
            self.attention_module  # Insert attention here
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Model output.
        """
        x = self.resnet.features(x)  # Pass through the feature extractor
        # x = self.global_pool(x)  # Apply global average pooling (reduce spatial dimensions to 1x1)
        x = self.global_avg_pool(x)  # GAP
        #x_max = self.gxobal_max_pool(x)  # Max Pool
        #x = torch.cat((x_avg, x_max), dim=1)  # Kết hợp
        x = self.classifier(x)  # Pass through the redefined Fully Connected layers
        return x


if __name__ == '__main__':
    from attention.CBAM import CBAMBlock
    from attention.BAM import BAMBlock
    from attention.scSE import scSEBlock

    # Initialize models with different attention mechanisms
    cbam_module = CBAMBlock(channel=256, reduction=16, kernel_size=7)
    model_cbam = ResNet18(pretrained=False, attention=cbam_module)

    bam_module = BAMBlock(channel=256, reduction=16, dia_val=2)
    model_bam = ResNet18(pretrained=False, attention=bam_module)

    scse_module = scSEBlock(channel=256)
    model_scse = ResNet18(pretrained=False, attention=scse_module)

    # Test input tensor
    x = torch.randn(32, 3, 224, 224)

    # Test models
    outputs_cbam = model_cbam(x)
    print("CBAM Output Shape:", outputs_cbam.shape)

    outputs_bam = model_bam(x)
    print("BAM Output Shape:", outputs_bam.shape)

    outputs_scse = model_scse(x)
    print("scSE Output Shape:", outputs_scse.shape)

    # Test model without attention
    model_no_attention = ResNet18(pretrained=False)
    outputs_no_attention = model_no_attention(x)
    print("No Attention Output Shape:", outputs_no_attention.shape)
