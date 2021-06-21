import os
from typing import Optional, Tuple

import numpy as np
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms import transforms


class PlantVillageDatamodule(LightningDataModule):
    """
    LightningDataModule for PlantVillage dataset.
    Dataset consists of images of plant leaves and background images. Leaves can be healthy or have a disease.
    Images are RBG 256x256

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
            self,
            image_size: int = 256,
            image_margin: int = 16,
            data_dir: str = "data",
            train_val_test_split: Tuple = (0.8, 0.1, 0.1),
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            fraction_size: float = 1.0,
            fraction_end: bool = False,
            **kwargs,
    ):
        super().__init__()
        assert len(train_val_test_split) == 2 or len(
            train_val_test_split) == 3, "there should be only two or three splits"
        self.share_val_test = len(train_val_test_split) == 2
        assert np.abs(train_val_test_split[0] + train_val_test_split[1] - 1) <= 1e-8 if self.share_val_test else np.abs(
            train_val_test_split[0] + train_val_test_split[1] + train_val_test_split[
                2] - 1) <= 1e-8, "the splits should add up to 1"

        self.data_dir = data_dir + "Plant_leave_diseases_dataset_with_augmentation/"
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.fraction_size = fraction_size
        self.fraction_end = fraction_end

        self.transforms = transforms.Compose(
            [transforms.Resize(image_size + image_margin), transforms.CenterCrop(image_size), transforms.ToTensor()]
        )

        # self.dims is returned when you call datamodule.size()
        self.dims = (3, image_size, image_size)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 39

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        dataset = ImageFolder(self.data_dir, transform=self.transforms)

        targets = dataset.targets
        self.classes = dataset.classes
        all_idx = np.arange(len(targets))
        if self.fraction_size < 1:
            first_idx, second_idx = train_test_split(
                all_idx,
                test_size=self.fraction_size if self.fraction_end else 1.0 - self.fraction_size,
                shuffle=True,
                stratify=targets)
            if self.fraction_end:
                all_idx = second_idx
            else:
                all_idx = first_idx

        all_targets = [targets[idx] for idx in all_idx]
        train_idx, test_idx = train_test_split(
            all_idx,
            test_size=self.train_val_test_split[1],
            shuffle=True,
            stratify=all_targets)
        if not self.share_val_test:
            targets_test = [targets[idx] for idx in test_idx]
            test_idx, val_idx = train_test_split(
                test_idx,
                test_size=self.train_val_test_split[2] / (1 - self.train_val_test_split[1]),
                shuffle=True,
                stratify=targets_test)
        else:
            val_idx = test_idx
        # Warp into Subsets and DataLoaders
        print(len(train_idx), len(test_idx), len(val_idx))
        self.data_train = Subset(dataset, train_idx)
        self.data_test = Subset(dataset, test_idx)
        self.data_val = Subset(dataset, val_idx)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
