import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform

class wildSet(Dataset):
    def __init__(self, file_path=None, labels=None, transform=None):
        self.file_path = file_path
        self.labels = labels
        self.transform = transform
    def __len__(self):
        if self.labels is None:
            return len(self.file_path)
        else:
            return len(self.labels)
    def __getitem__(self, idx):
        img_name = self.file_path.iloc[idx]
        try:
            image = io.imread(img_name)
            if self.transform:
                image = self.transform(image)
                label = self.labels.iloc[idx]
            return (image, label)
        except (OSError, ValueError):

            # print('OSERRoR!')
            print(f"FP: {self.file_path.iloc[idx]}, img_n: {img_name}")
            return None, None
