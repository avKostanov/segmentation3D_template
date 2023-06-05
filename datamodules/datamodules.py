from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from augmentations import make_augmentation_pipe
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class Segmentation3D_Dataset(Dataset):
    def __init__(self, dataset_path: Path, metadata: pd.DataFrame) -> None:
        self.dataset_path = dataset_path
        self.metadata = metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index) -> dict:
        sample = self.metadata.iloc[index].to_dict()

        image = np.load(self.dataset_path / sample['image_path']) / 255.
        image = image.astype(np.float32)
        mask = np.load(self.dataset_path / sample['mask_path'])
        mask = mask.astype(np.float32)

        return {
            'image': torch.from_numpy(image),
            'mask': torch.from_numpy(mask),
            'index': index
        }


class Segmentation3D_DataModule(LightningDataModule):
    def __init__(self, config: dict) -> None:
        self.config = config
        self.dataset_path = self.config['data']['dataset_path']
        self.augmentations = make_augmentation_pipe(prob=self.config['data']['augmentation_probability'])
        # TODO: add fixed data split

    def setup(self, stage: str) -> None:
        match stage:
            case 'fit':
                self.train_dataset = Segmentation3D_Dataset(
                    dataset_path=self.dataset_path,
                    metadata=pd.read_csv(self.dataset_path / 'train.csv')
                )
                self.val_dataset = Segmentation3D_Dataset(
                    dataset_path=self.dataset_path,
                    metadata=pd.read_csv(self.dataset_path / 'test.csv')
                )
            case 'test':
                self.test_dataset = Segmentation3D_Dataset(
                    dataset_path=self.dataset_path,
                    metadata=pd.read_csv(self.dataset_path / 'test.csv')
                )

    def train_dataloader(self):
        return DataLoader(self.train)
