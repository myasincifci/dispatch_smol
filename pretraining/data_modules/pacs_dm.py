import os
from typing import List

from torch import tensor
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torchvision import transforms as T
import pytorch_lightning as pl
from lightly.transforms.byol_transform import (BYOLTransform,
                                               BYOLView1Transform,
                                               BYOLView2Transform)
from lightly.transforms.utils import IMAGENET_NORMALIZE
from data_modules.image_dataset import ImageDataset

import json

class PacsDM(pl.LightningDataModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.data_dir = os.path.join(cfg.data_path, 'PACS')
        self.batch_size = cfg.param.batch_size

        self.train_transform = BYOLTransform(
            view_1_transform=T.Compose([
                T.Resize(96),
                BYOLView1Transform(input_size=96, gaussian_blur=0.0),
            ]),
            view_2_transform=T.Compose([
                T.Resize(96),
                BYOLView2Transform(input_size=96, gaussian_blur=0.0),
            ])
        )

        self.val_transform = T.Compose([
            T.Resize(96),
            T.ToTensor(),
            T.Normalize(
                mean=IMAGENET_NORMALIZE["mean"],
                std=IMAGENET_NORMALIZE["std"],
            ),
        ])

        train_split_path = '/home/yasin/repos/dispatch_smol/pretraining/data_modules/train.json'
        test_split_path = '/home/yasin/repos/dispatch_smol/pretraining/data_modules/test.json'
        with open(train_split_path) as f:
            train_set_map = json.load(f)
            f.close()
        with open(test_split_path) as f:
            test_set_map = json.load(f)
            f.close()

        self.classes = {
            'dog': 0, 
            'giraffe': 1, 
            'guitar': 2, 
            'house': 3, 
            'person': 4, 
            'horse': 5, 
            'elephant': 6
        }

        self.domains = {
            'sketch': 0, 
            'cartoon': 1, 
            'art_painting': 2, 
            'photo': 3
        }

        self.train_set = ImageDataset(self.data_dir, train_set_map, transform=self.train_transform, classes=self.classes, domains=self.domains)
        self.test_set = ImageDataset(self.data_dir, test_set_map, transform=self.val_transform, classes=self.classes, domains=self.domains)

        self.train_set_knn = ImageDataset(self.data_dir, train_set_map, transform=self.val_transform, classes=self.classes, domains=self.domains)
        self.test_set_knn = ImageDataset(self.data_dir, test_set_map, transform=self.val_transform, classes=self.classes, domains=self.domains)

        self.grouper = None

        self.cfg = cfg
        self.domain_mapper = None
        self.num_classes = len(self.classes)

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
            persistent_workers=True,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:    
        train_loader_knn = DataLoader(
            self.train_set_knn,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True
        )

        val_loader_knn = DataLoader(
            self.test_set_knn,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True
        )

        return [
            train_loader_knn,
            val_loader_knn
        ]
    
def main():
    from omegaconf import DictConfig

    cfg = DictConfig({
        'data_path': '/data/PACS',
        'param': {
            'batch_size': 32
        },

    })

    dm = PacsDM(cfg)
    train_loader = dm.train_dataloader()

    batch = next(iter(train_loader))
    a=1

if __name__ == '__main__':
    main()