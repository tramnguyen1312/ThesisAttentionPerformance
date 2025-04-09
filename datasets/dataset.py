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

        # # Define transforms
        # self.transform = transforms.Compose([
        #     # transforms.Resize((self.image_size, self.image_size)),  # Resize ảnh về 224x224
        #     transforms.RandomRotation(degrees=15),
        #     transforms.ToTensor(),
        #     transforms.RandomResizedCrop(size=(self.image_size, self.image_size), scale=(0.8, 1.0)),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        # ])
        # Define transforms
        if self.data_type == "train":
            # Transform cho train: gồm augmentations
            self.transform = transforms.Compose([
                transforms.Resize(size=(self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.0)),
                transforms.Pad(padding=30, fill=0),
                transforms.RandomCrop(size=(self.image_size, self.image_size)),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            # Transform cho test: chỉ resize và normalize
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),  # Resize ảnh về kích thước cố định
                transforms.ToTensor(),  # Chuyển ảnh từ PIL -> Tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Chuẩn hoá
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

    def print_indices(self):
        """
        In toàn bộ chỉ số của dataset (train hoặc test).
        """
        print(f"Dataset type: {self.data_type}")
        print(f"Total indices: {len(self.indices)}")
        print("Indices:", self.indices)

    def plot_random_images(self, num_images=5):
        """
        Hiển thị các ảnh ngẫu nhiên từ tập train/test.
        :param num_images: Số lượng ảnh cần hiển thị. Mặc định là 5.
        """
        num_total_images = len(self.indices)
        if num_total_images < num_images:
            print(f"Warning: Dataset only contains {num_total_images} images. Showing all.")
            num_images = num_total_images

            # Lấy các chỉ số ngẫu nhiên trong khoảng [0, len(self.indices)]
        random_indices = random.sample(range(num_total_images), num_images)

        # # Chọn ngẫu nhiên các chỉ mục ảnh cần hiển thị
        # random_indices = random.sample(self.indices, num_images)

        # Thiết lập grid để hiển thị với matplotlib
        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))  # Grid hiển thị
        mean = [0.485, 0.456, 0.406]  # Giá trị mean của Normalize
        std = [0.229, 0.224, 0.225]  # Giá trị std của Normalize

        for i, idx in enumerate(random_indices):
            print(idx)
            # Lấy ảnh và nhãn từ dataset
            actual_idx = self.indices[idx]  # Chỉ mục thực tế trong `self.full_dataset`
            image, label = self[actual_idx]

            # Đảo ngược Normalize (denormalize) để hiển thị ảnh
            denormalized_image = denormalize(image, mean, std)
            # Hiển thị ảnh
            ax = axes[i]
            ax.imshow(denormalized_image)
            ax.set_title(f"Label: {label}")
            ax.axis("off")

        plt.tight_layout()  # Sắp xếp các ảnh trong grid
        plt.show()


def plot_random_images(dataset, num_images=5):
    """
    Hiển thị các ảnh ngẫu nhiên từ dataset.
    :param dataset: Đối tượng dataset, kiểu GeneralDataset.
    :param num_images: Số lượng ảnh cần hiển thị.
    """
    # Kiểm tra nếu số lượng ảnh trong tập dữ liệu nhỏ hơn `num_images`
    if num_images > len(dataset.indices):
        print(
            f"Warning: Requested {num_images} images, but dataset only has {len(dataset.indices)}. Adjusting to {len(dataset.indices)}.")
        num_images = len(dataset.indices)

    # Chọn ngẫu nhiên các chỉ mục ảnh từ tập
    random_indices = random.sample(dataset.indices, num_images)

    # Thiết lập grid matplotlib
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    # Giá trị mean và std từ Normalize
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for i, idx in enumerate(random_indices):
        # Lấy ảnh và nhãn qua __getitem__ của dataset
        image, label = dataset[idx]

        # Đảo ngược Normalize để hiển thị
        denormalized_image = denormalize(image, mean, std)

        # Hiển thị ảnh
        ax = axes[i]
        ax.imshow(denormalized_image)
        ax.set_title(f"Label: {label}")
        ax.axis("off")

    plt.tight_layout()
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
    #caltech101_test.print_indices()
    caltech101_train.plot_random_images(num_images=3)

    # Load một batch dữ liệu
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Images shape: {images.shape}, Labels shape: {labels.shape}")
        break
