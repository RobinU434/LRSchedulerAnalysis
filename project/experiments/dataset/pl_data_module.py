import torch
import torchvision.transforms as transforms
from experiments import DATASET_PATH
from lightning import seed_everything
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


class CIFARLitDataModule(LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
        self.train_set: torch.utils.data.Dataset
        self.val_set: torch.utils.data.Dataset
        self.test_set: torch.utils.data.Dataset

        self.test_transform: transforms.Compose
        self.train_transform: transforms.Compose

    def setup(self, stage: str | None = None):
        dataset = CIFAR10(root=DATASET_PATH, train=True, download=True)
        data_means = (dataset.data / 255.0).mean(axis=(0, 1, 2))
        data_std = (dataset.data / 255.0).std(axis=(0, 1, 2))

        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(data_means, data_std),
            ]
        )
        # For training, we add some augmentation. Networks are too powerful and would overfit.
        self.train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(
                    (32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)
                ),
                transforms.ToTensor(),
                transforms.Normalize(data_means, data_std),
            ]
        )


        train_dataset = CIFAR10(
            root=DATASET_PATH,
            train=True,
            transform=self.train_transform,
            download=True,
        )
        val_dataset = CIFAR10(
            root=DATASET_PATH,
            train=True,
            transform=self.test_transform,
            download=True,
        )
        seed_everything(42)
        self.train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
        seed_everything(42)
        _, self.val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])

        # Loading the test set
        self.test_set = CIFAR10(
            root=DATASET_PATH, train=False, transform=self.test_transform, download=True
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=128,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=4,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=2048,
            shuffle=False,
            drop_last=False,
            num_workers=4,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=2048,
            shuffle=False,
            drop_last=False,
            num_workers=4,
        )
