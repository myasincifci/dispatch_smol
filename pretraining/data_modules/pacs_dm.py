import os
from typing import List

from torch import tensor
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
import pytorch_lightning as pl
from lightly.transforms.byol_transform import (BYOLTransform,
                                               BYOLView1Transform,
                                               BYOLView2Transform)
from lightly.transforms.utils import IMAGENET_NORMALIZE
from data_modules.pacs_h5_dataset import PACSDataset, BYOLViewTransform
from utils import DomainMapper

class PacsDM(pl.LightningDataModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.data_dir = cfg.data_path
        self.batch_size = cfg.param.batch_size

        self.train_transform = BYOLTransform(
            view_1_transform=T.Compose([
                T.Resize(96),
                BYOLViewTransform(
                    input_size=96
                ),
            ]),
            view_2_transform=T.Compose([
                T.Resize(96),
                BYOLViewTransform(
                    input_size=96,
                    gaussian_blur=0.1,
                    solarization_prob=0.2
                ),
            ])
        )

        self.val_transform = T.Compose([
            T.Normalize(
                mean=IMAGENET_NORMALIZE["mean"],
                std=IMAGENET_NORMALIZE["std"],
            ),
        ])

        self.train_set = PACSDataset(cfg.data_path, train=True, transform=self.train_transform)
        self.test_set = PACSDataset(cfg.data_path, train=False, transform=self.val_transform)

        self.train_set_knn = PACSDataset(cfg.data_path, train=True, transform=self.val_transform)
        self.test_set_knn = PACSDataset(cfg.data_path, train=False, transform=self.val_transform)

        self.grouper = None

        self.cfg = cfg
        self.domain_mapper = DomainMapper().setup(
            tensor(list(self.train_set.domains.values()))
        )
        self.num_classes = self.train_set.n_classes

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            pass

        elif stage == 'test':
            pass
        
        elif stage == 'predict':
            pass

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=lambda x: x
        )
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:    
        train_loader_knn = DataLoader(
            self.train_set_knn,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=lambda x: x
        )

        val_loader_knn = DataLoader(
            self.test_set_knn,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=lambda x: x
        )

        return [
            train_loader_knn,
            val_loader_knn
        ]
    
def main():
    pass 

if __name__ == '__main__':
    main()