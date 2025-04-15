import torch
import torchvision.transforms as transforms
from torchvision.datasets import Caltech101, Caltech256, OxfordIIITPet, STL10
from torch.utils.data import DataLoader, Subset
import ssl
from sklearn.model_selection import train_test_split

import random
import matplotlib.pyplot as plt
import numpy as np


def denormalize(image_tensor, mean, std):
    """
    Đảo ngược Normalize để chuyển ảnh về dải giá trị ban đầu ([0, 1] hoặc [0, 255]).
    :param image_tensor: Ảnh dạng tensor (C, H, W).
    :param mean: Danh sách giá trị mean sử dụng trong Normalize.
    :param std: Danh sách giá trị std sử dụng trong Normalize.
    :return: Ảnh dạng numpy array (H, W, C) dùng để plot.
    """
    image_tensor = image_tensor.clone()  # Tạo bản sao để tránh ảnh hưởng tensor gốc
    for t, m, s in zip(image_tensor, mean, std):  # Áp dụng đảo ngược Normalize cho từng kênh
        t.mul_(s).add_(m)  # t = t * std + mean
    image_array = image_tensor.detach().numpy()  # Chuyển tensor thành numpy
    image_array = np.transpose(image_array, (1, 2, 0))  # Chuyển trục từ (C, H, W) -> (H, W, C)
    image_array = np.clip(image_array, 0, 1)  # Giới hạn giá trị trong [0, 1] để hiển thị ảnh
    return image_array


import gdown
import os
import zipfile
import ssl


class DatasetDownloader:
    def __init__(self, dataset_name, output_dir="./datasets"):
        """
        Tải dữ liệu Caltech101 hoặc Caltech256 từ Google Drive nếu chưa có sẵn.
        :param dataset_name: Tên dataset, có thể là 'Caltech101' hoặc 'Caltech256'.
        :param output_dir: Thư mục lưu trữ dữ liệu.
        """
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.url = self.get_download_url()

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.dataset_path = os.path.join(self.output_dir, f"{self.dataset_name}.zip")

    def get_download_url(self):
        """
        Trả về URL tải xuống từ Google Drive cho dataset.
        :return: URL tải xuống của dataset.
        """
        if self.dataset_name == "Caltech101":
            return "https://drive.google.com/file/d/1lmgZZ2QdDxXiXyrkNzzjvwb6GDwFplLN/view?usp=sharing"
        elif self.dataset_name == "Caltech256":
            return "https://drive.google.com/uc?id=1hOjkjTkqfoZDra5MhOXoxPRdS1Xp9gZA/view?usp=sharing"
        else:
            raise ValueError("Dataset không hợp lệ. Chỉ hỗ trợ 'Caltech101' và 'Caltech256'.")

    def download(self):
        """
        Tải dataset từ Google Drive về.
        """
        if not os.path.exists(self.dataset_path):
            print(f"Downloading {self.dataset_name} dataset...")
            gdown.download(self.url, self.dataset_path, quiet=False, fuzzy=True)
        else:
            print(f"{self.dataset_name} dataset already exists.")

    def extract(self):
        """
        Giải nén file ZIP sau khi tải xong.
        """
        if self.dataset_path.endswith(".zip"):
            print(f"Extracting {self.dataset_name} dataset...")
            with zipfile.ZipFile(self.dataset_path, 'r') as zip_ref:
                zip_ref.extractall(self.output_dir)
        else:
            print(f"File {self.dataset_name}.zip không tồn tại hoặc không hợp lệ.")

    def download_and_extract(self):
        """
        Tải về và giải nén dataset nếu chưa có sẵn.
        """
        self.download()
        self.extract()
        print(f"{self.dataset_name} dataset has been downloaded and extracted to {self.output_dir}")

class GeneralDataset:
    """
    General dataset class wrapper using torchvision.datasets.
    Handles datasets like Caltech101. Automatically splits train/test.
    """

    def __init__(self, data_type, dataset_name, image_size=224, image_path="", test_split=0.2, random_seed=42):
        self.data_type = data_type  # 'train' hoặc 'test'
        self.dataset_name = dataset_name
        self.image_size = image_size
        self.image_path = image_path
        self.test_split = test_split  # Tỷ lệ dành cho tập test
        self.random_seed = random_seed

        if self.dataset_name in ["Caltech101", "Caltech256"]:
            self.downloader = DatasetDownloader(dataset_name=self.dataset_name, output_dir=self.image_path)
            self.downloader.download_and_extract()

        # # Define transforms

        common_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if self.data_type == "train":
            self.transform = transforms.Compose([
                common_transform,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=(-25, 25)),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2)),
            ])
        else:
            self.transform = common_transform

        # Load entire dataset
        self.full_dataset = self._load_dataset()

        self.num_classes = self._get_num_classes()

        # Perform train/test split
        self.indices = self._split_dataset()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        actual_idx = self.indices[idx]
        # Load ảnh và nhãn
        image, label = self.full_dataset[actual_idx]
        # Nếu ảnh không phải RGB, bỏ qua (có thể xuất hiện ảnh grayscale)
        if image.mode != "RGB":
            raise ValueError("Encountered a non-RGB image, which is unsupported.")

        # Áp dụng transform
        image = self.transform(image)

        return image, label

    def _load_dataset(self):
        ssl._create_default_https_context = ssl._create_unverified_context
        if self.dataset_name == ("Caltech101"):
            return Caltech101(root=self.image_path, download=True, transform=None)
        if self.dataset_name == ("Caltech256"):
             return Caltech256(root=self.image_path, download=True, transform=None)
        elif self.dataset_name == "STL10":
            split = "unlabeled" if self.data_type == "unlabeled" else self.data_type
            return STL10(root=self.image_path, split=split, download=True, transform=None)
        elif self.dataset_name == "Oxford-IIIT Pets":
            return OxfordIIITPet(root=self.image_path, download=True, transform=None)
        else:
            raise ValueError(f"Dataset {self.dataset_name} is not supported!")

    def _load_custom_dataset(self, dataset_dir):
        """
        Hàm tùy chỉnh để tải dataset Caltech101 hoặc Caltech256 từ thư mục giải nén.
        """
        from torchvision import datasets
        from torchvision.transforms import ToTensor

        # Kiểm tra nếu thư mục dataset đã tồn tại
        if not os.path.exists(dataset_dir):
            raise ValueError(f"Dataset directory {dataset_dir} không tồn tại!")
        return datasets.ImageFolder(root=dataset_dir, transform=ToTensor())

    def _split_dataset(self):
        # Lọc các chỉ số của ảnh RGB từ toàn bộ dataset
        all_indices = [
            idx for idx in range(len(self.full_dataset))
            if self.full_dataset[idx][0].mode == "RGB"
        ]

        # Train/Test Split
        train_indices, test_indices = train_test_split(
            all_indices, test_size=self.test_split, random_state=self.random_seed
        )

        if self.data_type == "train":
            return train_indices
        elif self.data_type == "test":
            return test_indices
        else:
            raise ValueError(f"Invalid data_type: {self.data_type}. Must be 'train' or 'test'.")

    def _get_num_classes(self):
        """
        Get the number of classes in the dataset by checking unique labels.
        """
        all_labels = [self.full_dataset[idx][1] for idx in range(len(self.full_dataset))]
        unique_labels = set(all_labels)  # Lấy danh sách các nhãn duy nhất
        return len(unique_labels)  # Tổng số lớp

    def print_indices(self):
        """
        In toàn bộ chỉ số của dataset (train hoặc test).
        """
        print(f"Dataset type: {self.data_type}")
        print(f"Total indices: {len(self.indices)}")
        print("Indices:", self.indices)

    def plot_random_images(self, num_images=5):
        """
        Hiển thị các ảnh ngẫu nhiên từ tập train/test, mỗi dòng tối đa 5 ảnh.
        :param num_images: Số lượng ảnh cần hiển thị.
        """
        num_total_images = len(self.indices)
        if num_total_images < num_images:
            print(f"Warning: Dataset only contains {num_total_images} images. Showing all.")
            num_images = num_total_images

            # Chọn ngẫu nhiên các chỉ mục ảnh
        random_indices = random.sample(range(num_total_images), num_images)

        # Thiết lập số cột và số dòng trong grid
        num_cols = 5  # Mỗi dòng tối đa 5 ảnh
        num_rows = (num_images + num_cols - 1) // num_cols  # Tính số dòng cần thiết

        # Thiết lập grid để hiển thị với matplotlib
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
        mean = [0.485, 0.456, 0.406]  # Mean trong Normalize
        std = [0.229, 0.224, 0.225]  # Std trong Normalize

        # Flatten `axes` để duyệt dễ dàng (nó sẽ là mảng 2D nếu `num_rows > 1`)
        axes = axes.flatten()

        # Hiển thị ảnh lên các subplot
        for i, idx in enumerate(random_indices):
            # Lấy ảnh và nhãn từ dataset
            image, label = self[idx]

            # Đảo ngược Normalize để hiển thị ảnh
            denormalized_image = denormalize(image, mean, std)

            # Truyền ảnh và nhãn vào subplot
            ax = axes[i]
            ax.imshow(denormalized_image)
            #ax.set_title(f"Label: {label}")
            ax.axis("off")

            # Ẩn các ô thừa (trường hợp số ảnh không chia hết cho số cột)
        for j in range(num_images, len(axes)):
            axes[j].axis("off")

            # Gọn layout và hiển thị
        plt.tight_layout()
        plt.show()


def save_random_images(dataset, num_images=10, output_dir="./random_images"):
    """
    Lưu ngẫu nhiên `num_images` ảnh từ dataset với nhãn tương ứng.
    :param dataset: Đối tượng dataset, kiểu GeneralDataset.
    :param num_images: Số lượng ảnh cần lưu.
    :param output_dir: Thư mục lưu trữ ảnh.
    """
    # Tạo thư mục lưu trữ nếu chưa tồn tại
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Kiểm tra nếu số lượng ảnh trong tập dữ liệu nhỏ hơn `num_images`
    if num_images > len(dataset.indices):
        print(
            f"Warning: Requested {num_images} images, but dataset only has {len(dataset.indices)}. Adjusting to {len(dataset.indices)}."
        )
        num_images = len(dataset.indices)

    # Chọn ngẫu nhiên các chỉ mục ảnh từ tập
    random_indices = random.sample(range(len(dataset.indices)), num_images)

    # Giá trị mean và std từ Normalize
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for idx in random_indices:
        # Lấy ảnh và nhãn qua __getitem__ của dataset
        image, label = dataset[idx]

        # Đảo ngược Normalize để lưu ảnh gốc
        denormalized_image = denormalize(image, mean, std)

        # Chuyển ảnh về dải giá trị từ [0, 1] -> [0, 255] và kiểu uint8
        denormalized_image = (denormalized_image * 255).astype(np.uint8)

        # Tạo tên file
        filename = os.path.join(output_dir, f"label_{label}_idx_{idx}.jpg")

        # Lưu ảnh
        plt.imsave(filename, denormalized_image)
        print(f"Saved: {filename}")

    print(f"Saved {num_images} random images to {output_dir}")


if __name__ == '__main__':
    # Tạo dataset
    caltech101_train = GeneralDataset(
        data_type="train",
        dataset_name="Caltech101",
        image_size=224,
        image_path="./datasets",
    )
    caltech101_test = GeneralDataset(
        data_type="test",
        dataset_name="Caltech101",
        image_size=224,
        image_path="./datasets",
    )
    print(f"Total images in the train dataset: {len(caltech101_train)}")
    print(f"Total images in the test dataset: {len(caltech101_test)}")
    # Tạo DataLoader cho Caltech101
    train_loader = DataLoader(
        caltech101_train,
        batch_size=32,
        shuffle=True,  # Shuffle dataset
        num_workers=0  # Để tránh lỗi đa luồng
    )
    train_loader = DataLoader(
        caltech101_test,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )
    # caltech101_test.print_indices()
    caltech101_test.plot_random_images(num_images=20)

    # Load một batch dữ liệu
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Images shape: {images.shape}, Labels shape: {labels.shape}")
        break
    # save_random_images(caltech101_test, num_images=10, output_dir="./images")

