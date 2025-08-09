
from attention.CBAM import CBAMBlock
from attention.BAM import BAMBlock
from attention.scSE import scSEBlock
from backbone import ResNet18, VGG16

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from datasets import GeneralDataset
from torch.utils.data import DataLoader
import os
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image

from tqdm import tqdm
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Đảo ngược Normalize: Tensor -> Tensor đã về đúng màu
    """
    tensor = tensor.clone().cpu()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

def get_top_correct_images_by_confidence(model, dataloader, device, dataset, num_samples=10, save_txt="top_correct_paths.txt",save_dir="temp"):
    model.eval()
    all_correct = []
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for batch_idx, (images, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Đánh giá mô hình"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            correct = preds.eq(labels)

            for i in range(images.size(0)):
                if correct[i]:
                    confidence = probs[i][preds[i]].item()
                    img_tensor = images[i].cpu()
                    label = labels[i].item()
                    pred = preds[i].item()

                    # Lưu ảnh thực tế vào thư mục tạm
                    file_name = f"img_{batch_idx}_{i}_true{label}_pred{pred}.jpg"
                    save_path = os.path.join(save_dir, file_name)
                    to_pil_image(denormalize(img_tensor)).save(save_path)

                    all_correct.append((confidence, save_path))

    top_correct = sorted(all_correct, key=lambda x: x[0], reverse=True)[:num_samples]

    with open(save_txt, "w") as f:
        for conf, path in top_correct:
            f.write(f"{conf:.4f} {path}\n")
    return [p for _, p in top_correct]

def apply_gradcam_to_paths(model,
                           paths,
                           target_layers_dict,
                           save_dir="gradcam_output",
                           target_class=1,
                           image_size=(224, 224)):
    """
    Áp dụng Grad-CAM cho mỗi ảnh trong `paths`.
    Với mỗi (label, layers) trong `target_layers_dict`, vẽ Grad-CAM map và ghi nhãn plot theo label.

    :param model:               Mô hình đã load weights và ở chế độ .eval()
    :param paths:               List các đường dẫn đến ảnh
    :param target_layers_dict:  Dict[str, List[module]]; key là label, value là list các layer để GradCAM
    :param save_dir:            Thư mục lưu kết quả
    :param target_class:        Chỉ số lớp dùng làm target trong CAM
    :param image_size:          Kích thước resize trước khi feed vào mô hình
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    for path in tqdm(paths, desc="Grad-CAM cho ảnh đúng"):
        img_pil = Image.open(path).convert("RGB")
        rgb_img = np.array(img_pil.resize(image_size)).astype(np.float32) / 255.0
        input_tensor = transform(img_pil).unsqueeze(0).to(device)

        # thu thập kết quả Grad-CAM theo từng label
        cam_results = {}
        for label, layers in target_layers_dict.items():
            cam = GradCAMPlusPlus(model=model, target_layers=layers)
            cam_map = cam(input_tensor=input_tensor,
                          targets=[ClassifierOutputTarget(target_class)])[0]
            vis = show_cam_on_image(rgb_img, cam_map, use_rgb=True)
            cam_results[label] = vis

        # Vẽ ảnh gốc và các Grad-CAM map
        n = 1 + len(cam_results)
        plt.figure(figsize=(4 * n, 4))
        plt.subplot(1, n, 1)
        plt.imshow(img_pil)
        plt.title("Gốc")
        plt.axis("off")

        for i, (label, vis) in enumerate(cam_results.items(), start=2):
            plt.subplot(1, n, i)
            plt.imshow(vis)
            plt.title(label)
            plt.axis("off")

        plt.tight_layout()
        fname = os.path.basename(path).replace(".jpg", "_gradcam.png")
        plt.savefig(os.path.join(save_dir, fname))
        plt.close()
def get_all_image_paths(root_path):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    image_paths = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        for file in filenames:
            if file.lower().endswith(image_extensions):
                full_path = os.path.join(dirpath, file)
                image_paths.append(full_path)
    return image_paths
def main():
    # Cấu hình
    model_path = "/Users/minhbui/Personal/Project/Katalyst/ThesisAttentionPerformance/model/VGG16/isic_v2.pth"
    root = "/Users/minhbui/Personal/Project/Katalyst/ThesisAttentionPerformance/datasets/datasets"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 7
    num_top_images = 40
    #
    # dataset = GeneralDataset('ICIS2018', root)
    # train_dataset, test_dataset = dataset.get_splits(val_size=0.2, seed=42, image_size=224)
    #
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=64,
    #     shuffle=False,
    #     num_workers=0
    # )

    # Load mô hình
    model = VGG16(pretrained=False, num_classes=num_classes, attn_type="scSE")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Lấy top N ảnh đúng nhất



    # correct_paths = get_top_correct_images_by_confidence(
    #     model=model,
    #     dataloader=test_loader,
    #     device=device,
    #     dataset=test_dataset,
    #     num_samples=num_top_images,
    #     save_txt="correct_paths.txt",
    #     save_dir="temp"
    # )

    # target_layers_dict = {
    #     "Trước GLSA": [model.stem[6][1]],
    #     "Sau GLSA": [model.mha_block],
    #     "Layer4 cuối": [model.layer4[-1]]
    # }
    # 1) Xây dict label → list các module
    target_layers_dict = {
        "X": [model.mha_block.hook_x],
        "F1": [model.mha_block.hook_f1],
        #"F2": [model.mha_block.hook_f2],
        "F2": [model.mha_block.hook_res],
        "Out": [model.mha_block.hook_out],
    }

    correct_paths =get_all_image_paths('/Users/minhbui/Personal/Project/Katalyst/ThesisAttentionPerformance/datasets/datasets/ISIC2018/ISIC2018_Task3_Validation_Input/ISIC2018_Task3_Validation_Input')
    apply_gradcam_to_paths(
        model=model,
        paths=correct_paths,
        target_layers_dict=target_layers_dict,
        save_dir="temp/gradcam_labeled",
        target_class=1,
        image_size=(224, 224)
    )

if __name__ == "__main__":
    main()