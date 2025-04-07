import torch
import torchvision.transforms as transforms
from torchvision.datasets import Caltech101, Caltech256, OxfordIIITPet, STL10
from torch.utils.data import DataLoader, Subset
import ssl
from sklearn.model_selection import train_test_split

import random
import matplotlib.pyplot as plt
import numpy as np


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

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),  # Resize ảnh về 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

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
        if self.dataset_name == "Caltech101":
            return Caltech101(root=self.image_path, download=True, transform=None)
        elif self.dataset_name == "Caltech256":
            return Caltech256(root=self.image_path, download=True, transform=None)
        elif self.dataset_name == "STL10":
            split = "unlabeled" if self.data_type == "unlabeled" else self.data_type
            return STL10(root=self.image_path, split=split, download=True, transform=None)
        elif self.dataset_name == "Oxford-IIIT Pets":
            return OxfordIIITPet(root=self.image_path, download=True, transform=None)
        else:
            raise ValueError(f"Dataset {self.dataset_name} is not supported!")

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

    def plot_random_images(self, num_images=5):
        """
        Plot random images from the dataset.
        :param num_images: Number of random images to display. Default: 5
        """
        random_indices = random.sample(self.indices, num_images)  # Chọn ngẫu nhiên ảnh từ indices
        plt.figure(figsize=(15, 5))  # Kích thước của grid hiển thị
        for i, idx in enumerate(random_indices):
            image, label = self.full_dataset[idx]  # Lấy ảnh và nhãn gốc (chưa transform)

            # Khi sử dụng transform, cần đảo ngược Normalize để hiển thị ảnh
            if self.transform:
                transform_reverse = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor()
                ])
                image = np.array(image)  # Chuyển dữ liệu sang numpy để plot
            else:
                image = np.array(image)

                # Plot ảnh
            plt.subplot(1, num_images, i + 1)
            plt.imshow(image)  # Hiển thị ảnh
            plt.title(f"Label: {label}")
            plt.axis("off")

        plt.show()


if __name__ == '__main__':
    # Tạo dataset
    caltech101_train = GeneralDataset(
        data_type="train",
        dataset_name="STL10",
        image_size=224,
        image_path="./datasets",
    )
    caltech101_test = GeneralDataset(
        data_type="test",
        dataset_name="STL10",
        image_size=224,
        image_path="./datasets",
    )
    print(f"Total images in the train dataset: {len(caltech101_train)}")
    print(f"Total images in the test dataset: {len(caltech101_test)}")
    # Tạo DataLoader cho Caltech101
    train_loader = DataLoader(
        caltech101_train,
        batch_size=32,  # Batch size
        shuffle=True,  # Shuffle dataset
        num_workers=0  # Để tránh lỗi đa luồng
    )
    train_loader = DataLoader(
        caltech101_test,
        batch_size=32,  # Batch size
        shuffle=True,  # Shuffle dataset
        num_workers=0  # Để tránh lỗi đa luồng
    )

    caltech101_train.plot_random_images(num_images=5)

    # Load một batch dữ liệu
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Images shape: {images.shape}, Labels shape: {labels.shape}")
        break
