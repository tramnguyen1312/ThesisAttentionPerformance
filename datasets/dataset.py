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


# ----------------------- Utility Functions -----------------------
def set_seed(seed: int = 42):
    """
    Set seed for reproducibility across modules.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

        if not os.path.exists(self.zip_path):
            url = f"https://drive.google.com/uc?id={drive_id}"
            gdown.download(url, self.zip_path, quiet=False)

        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.output_dir)


# ----------------------- General Dataset Class -----------------------
class GeneralDataset(Dataset):
    """
    Wrapper for various datasets with stratified train/validation split.
    """
    def __init__(
        self,
        split: str,
        name: str,
        root: str,
        image_size: int = 224,
        val_size: float = 0.2,
        seed: int = 42,
    ):
        assert split in ['train', 'val'], "split must be 'train' or 'val'"

        self.split = split
        self.name = name
        self.root = root
        self.val_size = val_size

        # Download dataset if needed
        if name in DatasetDownloader.DRIVE_IDS:
            DatasetDownloader(name, root).download_and_extract()

        # Load all images and labels to memory
        self.images, self.labels = self._load_data()
        assert self.images, f"No data for {name} in {root}"

        # Stratified split
        indices = np.arange(len(self.labels))
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=val_size, random_state=seed
        )
        train_idx, val_idx = next(splitter.split(indices, self.labels))
        self.indices = train_idx if split == 'train' else val_idx

        # Define transforms
        self.transform = self._build_transform(image_size)

        self.num_classes = len(set(self.labels))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        actual_idx = self.indices[idx]
        image = self.images[actual_idx]
        label = self.labels[actual_idx]
        return self.transform(image), label

    def _build_transform(self, size: int):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if self.split == 'train':
            return transforms.Compose([
                transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((size,size)),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalize,
            ])

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

    def get_sampler(self) -> WeightedRandomSampler:
        """
        Create a weighted sampler to handle class imbalance.
        """
        counts = np.bincount([self.labels[i] for i in self.indices])
        weights = 1.0 / counts
        sample_weights = [weights[self.labels[i]] for i in self.indices]
        return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    def plot_random_images(self, num_images: int = 5):
        """
        Plot a grid of random images from the dataset split.
        """
        num_images = min(num_images, len(self.indices))
        choices = random.sample(list(self.indices), num_images)

        cols = min(5, num_images)
        rows = (num_images + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        axes = axes.flatten()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        for ax, idx in zip(axes, choices):
            img_t, lbl = self.__getitem__(self.indices.tolist().index(idx))
            ax.imshow(denormalize(img_t, mean, std))
            ax.set_title(str(lbl))
            ax.axis('off')

        for ax in axes[num_images:]:
            ax.axis('off')

        plt.tight_layout()
        plt.show()


# ----------------------- Usage Example -----------------------
if __name__ == '__main__':
    train_dataset = GeneralDataset('train', 'HAM10000', './datasets')
    val_dataset = GeneralDataset('val', 'HAM10000', './datasets')
    print(f"Total images in the train dataset: {len(train_dataset)}")
    print(f"Total images in the test dataset: {len(val_dataset)}")

    img, label = val_dataset[0]
    print("Val sample 0 – image shape:", img.shape, " label:", label)
    img, label = train_dataset[0]
    print("Train sample 0 – image shape:", img.shape, " label:", label)


    #val_dataset.plot_random_images(num_images=10)
