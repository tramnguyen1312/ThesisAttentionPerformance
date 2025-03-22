import torch
import torchvision.transforms as transforms
from PIL import Image
from attention.CBAM import CBAMBlock


def process_image_through_cbam(image_path, save_path=None):
    # Chuẩn bị CBAM block
    channel = 510  # Ví dụ, số channel trong kiến trúc bạn mong muốn (phải khớp với model)
    reduction = 16
    cbam = CBAMBlock(channel=channel, reduction=reduction, kernel_size=7)  # Kernel size mặc định là 7

    # Load ảnh gốc
    image = Image.open(image_path).convert('RGB')

    # Transform ảnh sang dạng tensor
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize ảnh về kích thước tiêu chuẩn 224x224
        transforms.ToTensor(),  # Chuyển ảnh thành tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa theo ImageNet
    ])
    input_tensor = preprocess(image)  # Đầu ra có kích thước (3, 224, 224)
    input_tensor = input_tensor.unsqueeze(0)  # Thêm batch size -> (1, 3, 224, 224)

    print(f"Input tensor shape: {input_tensor.shape}")

    # Fake đầu vào cho kênh CBAM (điều chỉnh số kênh cho đầu vào)
    input_tensor = input_tensor.repeat(1, channel // 3, 1, 1)  # Chuyển thành dạng (1, 512, 224, 224)

    # Chạy ảnh qua CBAM
    output_tensor = cbam(input_tensor)

    print(f"Output tensor shape: {output_tensor.shape}")

    # Chuyển tensor kết quả về ảnh
    output_image = output_tensor[0].mean(dim=0)  # Lấy trung bình theo chiều kênh -> (224, 224)
    output_image = output_image.detach().cpu().numpy()  # Chuyển sang numpy
    output_image = (output_image - output_image.min()) / (
                output_image.max() - output_image.min())  # Chuẩn hóa về [0, 1]
    output_image = (output_image * 255).astype('uint8')  # Chuẩn hóa về [0, 255]

    # Convert ndarray sang ảnh
    output_pil = Image.fromarray(output_image)

    # Lưu hoặc hiển thị ảnh kết quả
    if save_path:
        output_pil.save(save_path)
        print(f"Kết quả đã được lưu tại: {save_path}")
    else:
        output_pil.show()


if __name__ == "__main__":
    # Đường dẫn ảnh đầu vào
    image_path = "images/input/anh-cho-thumbnail.jpg"
    save_path = "images/output/anh-cho-thumbnail.jpg"  # Đường dẫn lưu ảnh kết quả (hoặc None nếu chỉ muốn hiển thị)

    # Gọi hàm xử lý
    process_image_through_cbam(image_path, save_path)