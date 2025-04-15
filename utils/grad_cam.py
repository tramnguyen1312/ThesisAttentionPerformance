import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from attention.CBAM import CBAMBlock
from attention.BAM import BAMBlock
from attention.scSE import scSEBlock
from backbone import ResNet18, VGG16


def load_model(model_path, attention=None):
    """
    Hàm load mô hình từ file .pth.
    :param model_path: Đường dẫn đến file mô hình .pth
    :param attention: Attention module (nếu có)
    :return: Model PyTorch đã được load
    """
    # Load mô hình ResNet50 với attention module tùy chọn
    model = ResNet18(pretrained=False, attention=attention)
    model.load_state_dict(torch.load(model_path))  # Load trọng số từ file .pth
    model.eval()  # Đặt model vào chế độ inference
    return model


def preprocess_image(image_path):
    """
    Hàm xử lý ảnh đầu vào (resize và chuẩn hóa) để đưa vào mô hình.
    :param image_path: Đường dẫn đến file ảnh.
    :return: Tensor ảnh đã được xử lý.
    """
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0)  # Thêm batch dimension
    return input_tensor


def visualize_gradcam(model_path, image_path, attention=None):

    model = load_model(model_path, attention)

    input_tensor = preprocess_image(image_path)

    target_layer = model.resnet.features[4]

    # Tạo Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
    grayscale_cam = cam(input_tensor=input_tensor)[0]  # CAM cho ảnh đầu tiên trong batch

    rgb_img = input_tensor.squeeze().permute(1, 2, 0).numpy()

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    plt.imshow(visualization)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    model_path = "/Users/minhbui/Personal/Project/Katalyst/ThesisAttentionPerformance/result/best_model.pth"
    image_path = "/Users/minhbui/Personal/Project/Katalyst/ThesisAttentionPerformance/datasets/images/label_0_idx_936.jpg"

    attention_module = scSEBlock(channel=2048, reduction=16, kernel_size=7)
    visualize_gradcam(model_path, image_path, attention=attention_module)
