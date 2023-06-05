from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from sampling import make_sampler
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
        return DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training']['num_workers'],
            pin_memory=True,
            sampler=(None if not self.config['training']['sampler'] else make_sampler(self.train_dataset.metadata))
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config['testing']['batch_size'],
            num_workers=self.config['testing']['num_workers'],
            pin_memory=True
        )
