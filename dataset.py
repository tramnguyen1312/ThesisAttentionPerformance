import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import STL10
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

class STL10Dataset(Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        super().__init__()
        split = 'train' if train else 'test'
        self.dataset = STL10(root=root, split=split, download=download)
        self.transform = transform
        self.categories = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
