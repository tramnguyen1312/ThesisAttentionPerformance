from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from attention.CBAM import CBAMBlock   # CBAM module
from attention.BAM import BAMBlock    # BAM module
from attention.scSE import scSEBlock  # scSE module

class ResNet50(torch.nn.Module):
    def __init__(self, pretrained=False, attention_type= None):
        super().__init__()
        # Load mô hình ResNet50 cơ bản
        self.resnet = ptcv_get_model("resnet50", pretrained=pretrained)

        if self.attention_type is not None:
            self.attention_type = attention_type.lower()
            self.input_attention_module = self._get_attention_module(channel=3)

    def _get_attention_module(self, channel):
        """
        Trả về Attention module theo loại đã chọn.
        Args:
            channel (int): Số lượng kênh đầu vào của Attention.
        Returns:
            torch.nn.Module: Attention module (CBAM, BAM, scSE).
        """
        if self.attention_type == 'cbam':
            return CBAMBlock(channel=channel, reduction=16, kernel_size=7)
        elif self.attention_type == 'bam':
            return BAMBlock(channel=channel, reduction=16, dia_val=2)
        elif self.attention_type == 'scse':
            return scSEBlock(channels=channel)
        else:
            raise ValueError(f"Unknown attention_type: {self.attention_type}")

    def forward(self, x):
        # Attention trước khi vào ResNet
        if self.use_attention:
            x = self.input_attention_module(x)

        # Qua ResNet
        x = self.resnet.features(x)
        x = x.view(x.size(0), -1)
        x = self.resnet.output(x)

        return x


if __name__ == '__main__':
    # Khởi tạo model với attention khác nhau
    model_cbam = ResNet50(pretrained=False, attention_type='cbam')
    model_bam = ResNet50(pretrained=False, attention_type='bam')
    model_scse = ResNet50(pretrained=False, attention_type='scse')

    x = torch.randn(2, 3, 224, 224)
    outputs = model_scse(x)
    print(outputs)