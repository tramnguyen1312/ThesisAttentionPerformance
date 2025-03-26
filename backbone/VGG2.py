import torch
from pytorchcv.model_provider import get_model as ptcv_get_model
from attention.CBAM import CBAMBlock
from attention.BAM import BAMBlock
from attention.scSE import scSEBlock


class VGG16(torch.nn.Module):
    def __init__(self, pretrained=False, attention_type=None):
        """
        Args:
            pretrained (bool): Nếu True, tải mô hình VGG16 đã pretrained.
            use_attention (bool): Nếu True, sử dụng Attention module.
            attention_type (str): Loại Attention được sử dụng (cbam, bam, scse).
        """
        super().__init__()
        self.vgg16 = ptcv_get_model("vgg16", pretrained=pretrained)
        self.attention_type = attention_type
        if self.attention_type is not None:
            self.attention_type = attention_type.lower()
            self.input_attention_module = self._get_attention_module(channel=3)

    def _get_attention_module(self, channel):
        """Trả về Attention module theo loại đã chọn"""
        if self.attention_type == 'cbam':
            return CBAMBlock(channel=channel, reduction=16, kernel_size=7)
        elif self.attention_type == 'bam':
            return BAMBlock(channel=channel, reduction=16, dia_val=2)
        elif self.attention_type == 'scse':
            return scSEBlock(channel)
        else:
            return None

    def forward(self, x):
        if self.attention_type is not None:
            x = self.input_attention_module(x)
        x = self.vgg16.features(x)
        x = x.view(x.size(0), -1)
        x = self.vgg16.output(x)

        return x


if __name__ == '__main__':

    model_no_attention = VGG16(pretrained=False)
    model_cbam = VGG16(pretrained=False, attention_type='cbam')
    model_bam = VGG16(pretrained=False, attention_type='bam')
    model_scse = VGG16(pretrained=False,  attention_type='scse')

    x = torch.randn(2, 3, 224, 224)
    outputs = model_cbam(x)
    print (model_cbam)
    print(outputs)