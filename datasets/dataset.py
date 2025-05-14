import os
import random
import ssl
import zipfile

import gdown
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import STL10, Caltech101, Caltech256, OxfordIIITPet
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def denormalize(image_tensor: torch.Tensor, mean: list, std: list) -> np.ndarray:
    """
    Denormalize a tensor image and convert to numpy array for plotting.
    """
    img = image_tensor.clone()
    for channel, m, s in zip(img, mean, std):
        channel.mul_(s).add_(m)
    array = img.numpy().transpose(1, 2, 0)
    return np.clip(array, 0, 1)


# ----------------------- Dataset Downloader -----------------------
class DatasetDownloader:
    """
    Download and extract datasets from Google Drive based on predefined IDs.
    """
    DRIVE_IDS = {
        "Caltech101": "1lmgZZ2QdDxXiXyrkNzzjvwb6GDwFplLN",
        "Caltech256": "1Ou7A5FmPH6vJ5l-syt7geZhnljes0KEV",
        "HAM10000": "1YgtSWc2tPP0qHIV-hf1qpJeLXAkmeV3O",
        "ISIC2018": "1G5xrbsVC-saor6LOmPJLePDIkO62YN1j",
    }

    def __init__(self, name: str, output_dir: str):
        self.name = name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.zip_path = os.path.join(output_dir, f"{name}.zip")

    def download_and_extract(self):
        drive_id = self.DRIVE_IDS.get(self.name)
        if not drive_id:
            return

        if not os.path.exists( os.path.join(self.output_dir, self.name)):
            if not os.path.exists(self.zip_path):
                url = f"https://drive.google.com/uc?id={drive_id}"
                gdown.download(url, self.zip_path, quiet=False)

            print(f"Extracting {self.name}...")
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                for member in tqdm(zip_ref.namelist(), desc=f"Extracting {self.name}", unit="file"):
                    # skip macOS metadata
                    if '__MACOSX' in member:
                        continue
                    # extract actual dataset files
                    zip_ref.extract(member, self.output_dir)
        else:
            print('Dataset đã tồn tại')


# ----------------------- General Dataset Class -----------------------
class GeneralDataset(Dataset):
    """
    Wrapper for various datasets with stratified train/validation split.
    """

    def __init__(
            self,
            name: str,
            root: str,
    ):
        self.name = name
        self.root = root

        # Download dataset if needed
        if name in DatasetDownloader.DRIVE_IDS:
            DatasetDownloader(name, root).download_and_extract()

        # Load all images and labels to memory
        self.images, self.labels = self._load_data()
        assert self.images, f"No data for {name} in {root}"

        self.num_classes = len(set(self.labels))

    def get_splits(
            self,
            val_size: float = 0.2,
            seed: int = 42,
            image_size: int = 224
    ):
        """
        Trả về (train_dataset, val_dataset) sau stratified split và transform.
        """
        labels = np.array(self.labels)
        idx = np.arange(len(labels))
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=val_size, random_state=seed
        )
        train_idx, val_idx = next(splitter.split(idx, labels))

        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
            transforms.RandomPerspective(0.2, p=0.2),
            transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ])
        val_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])

        class SubsetMemDataset(Dataset):
            def __init__(self, imgs, lbls, indices, tf):
                self.imgs = imgs
                self.lbls = lbls
                self.indices = indices
                self.tf = tf

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, i):
                img = self.imgs[self.indices[i]]
                lbl = self.lbls[self.indices[i]]
                return self.tf(img), lbl

        train_ds = SubsetMemDataset(self.images, self.labels, train_idx, train_tf)
        val_ds = SubsetMemDataset(self.images, self.labels, val_idx, val_tf)
        return train_ds, val_ds

    def _load_data(self):
        """
        Dispatch loading strategy based on dataset name.
        """
        if self.name in ['Caltech101', 'Caltech256', 'Oxford-IIIT Pets']:
            return self._load_standard_via_torchvision()
        if self.name == 'STL10':
            return self._load_stl10()
        if self.name == 'HAM10000':
            return self._load_ham10000()
        if self.name == 'isic-2018-task-3':
            return self._load_isic2018()

        raise ValueError(f"Unknown dataset: {self.name}")

    def _load_standard_via_torchvision(self):
        """
        Load Caltech101, Caltech256, or Oxford-IIIT Pets entirely into memory.
        """
        loader_map = {
            'Caltech101': Caltech101,
            'Caltech256': Caltech256,
            'Oxford-IIIT Pets': OxfordIIITPet,
        }
        ds_class = loader_map[self.name]
        ds = ds_class(root=self.root, download=(self.name != 'Caltech101'))

        images, labels = [], []
        for img, lbl in ds:
            images.append(img.convert('RGB'))
            labels.append(lbl)
        return images, labels

    def _load_stl10(self):
        """
        Load STL10 into memory.
        """
        ds = STL10(root=self.root, split='train', download=True)
        images, labels = [], []
        for img, lbl in ds:
            images.append(img.convert('RGB'))
            labels.append(lbl)
        return images, labels

    def _load_ham10000(self):
        """
        Load HAM10000 using metadata CSV and image folders.
        """
        candidates = [
            os.path.join(self.root, 'HAM10000_metadata.csv'),
            os.path.join(self.root, 'HAM10000', 'HAM10000_metadata.csv'),
        ]
        meta_path = next((p for p in candidates if os.path.exists(p)), None)
        assert meta_path, 'HAM10000 metadata missing'

        base_dir = os.path.dirname(meta_path)
        df = pd.read_csv(meta_path)
        label_map = {dx: idx for idx, dx in enumerate(df['dx'].unique())}

        dirs = [
            os.path.join(base_dir, 'HAM10000_images_part_1'),
            os.path.join(base_dir, 'HAM10000_images_part_2'),
            base_dir,
        ]

        images, labels = [], []
        for _, row in df.iterrows():
            filename = f"{row['image_id']}.jpg"
            for d in dirs:
                path = os.path.join(d, filename)
                if os.path.exists(path):
                    with Image.open(path) as img:
                        images.append(img.convert('RGB'))
                    labels.append(label_map[row['dx']])
                    break

        return images, labels

    def _load_isic2018(self):
        """
        Load ISIC 2018 Task 3 dataset for lesion classification.
        Recursively search self.root for metadata CSV and image folder.
        """
        # Find metadata CSV
        meta_path = None
        for rd, _, files in os.walk(self.root):
            if 'ISIC2018_Task3_Training_GroundTruth.csv' in files:
                meta_path = os.path.join(rd, 'ISIC2018_Task3_Training_GroundTruth.csv')
                break
        assert meta_path, f"ISIC2018 metadata CSV missing under {self.root}"
        df = pd.read_csv(meta_path)
        cols = df.columns[1:]

        # Find image folder
        img_dir = None
        for rd, dirs, _ in os.walk(self.root):
            if 'ISIC2018_Task3_Training_Input' in dirs:
                img_dir = os.path.join(rd, 'ISIC2018_Task3_Training_Input')
                break
        assert img_dir, f"ISIC2018 images folder missing under {self.root}"

        # Load
        imgs, lbls = [], []
        for _, row in df.iterrows():
            fn = 'ISIC2018_Task3_Training_Input/' +  row['image'] + '.jpg'
            p = os.path.join(img_dir, fn)
            #print(p)
            if os.path.exists(p):
                with Image.open(p) as im:
                    imgs.append(im.convert('RGB'))
                one = row[cols].values.astype(np.float32)
                lbls.append(int(np.argmax(one)))
        return imgs, lbls

    def get_sampler(self) -> WeightedRandomSampler:
        """
        Create a weighted sampler to handle class imbalance.
        """
        counts = np.bincount([self.labels[i] for i in self.indices])
        weights = 1.0 / counts
        sample_weights = [weights[self.labels[i]] for i in self.indices]
        return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

def plot_random_images(
    dataset,
    num_images: int = 5,
    mean: list = [0.485, 0.456, 0.406],
    std: list  = [0.229, 0.224, 0.225]
):
    """
    Vẽ ngẫu nhiên `num_images` ảnh từ `dataset` (PyTorch Dataset trả về (img_tensor, label)).
    """
    N = len(dataset)
    num_images = min(num_images, N)
    idxs = random.sample(range(N), num_images)

    cols = min(5, num_images)
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = axes.flatten()

    for ax, i in zip(axes, idxs):
        img_t, lbl = dataset[i]
        img = denormalize(img_t, mean, std)
        ax.imshow(img)
        ax.set_title(f"Label: {lbl}")
        ax.axis('off')

    for ax in axes[num_images:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_label_histogram(
    train_dataset,
    val_dataset,
    num_classes: int,
    title_train: str = 'Train Label Distribution',
    title_val: str = 'Validation Label Distribution',
    bar_width: float = 0.8
):
    """
    Vẽ histogram phân bố nhãn của train và validation datasets.
    Luôn hiển thị đủ cột cho tất cả nhãn; nếu số lớp > 15, không hiển thị tên nhãn (để tránh quá dày).
    """
    train_labels = [label for _, label in train_dataset]
    val_labels = [label for _, label in val_dataset]

    x = np.arange(num_classes)
    train_counts = np.bincount(train_labels, minlength=num_classes)
    val_counts = np.bincount(val_labels, minlength=num_classes)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Train
    axes[0].bar(x, train_counts, width=bar_width, edgecolor='black')
    axes[0].set_title(title_train)
    axes[0].set_xlabel('Label')
    axes[0].set_ylabel('Count')
    axes[0].set_xticks(x)
    if num_classes <= 15:
        axes[0].set_xticklabels(x)
    else:
        axes[0].set_xticklabels([''] * num_classes)

    # Validation
    axes[1].bar(x, val_counts, width=bar_width, edgecolor='black')
    axes[1].set_title(title_val)
    axes[1].set_xlabel('Label')
    axes[1].set_ylabel('Count')
    axes[1].set_xticks(x)
    if num_classes <= 15:
        axes[1].set_xticklabels(x)
    else:
        axes[1].set_xticklabels([''] * num_classes)

    plt.tight_layout()
    plt.show()

# ----------------------- Usage Example -----------------------
if __name__ == '__main__':
    dataset = GeneralDataset('Caltech101', './datasets')
    train_dataset, val_dataset = dataset.get_splits(val_size=0.2, seed=42, image_size=224)
    print(f"Total images in the train dataset: {len(train_dataset)}")
    print(f"Total images in the test dataset: {len(val_dataset)}")

    img, label = val_dataset[0]
    print("Val sample 0 – image shape:", img.shape, " label:", label)
    img, label = train_dataset[0]
    print("Train sample 0 – image shape:", img.shape, " label:", label)

    #plot_random_images(train_dataset, num_images=8)
    plot_random_images(val_dataset, num_images=25)

    plot_label_histogram(train_dataset, val_dataset, dataset.num_classes)

    # val_dataset.plot_random_images(num_images=10)
