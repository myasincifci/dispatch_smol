from typing import List

import torch
from torch.utils.data import Subset

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
import pytorch_lightning as pl
from lightly.transforms.byol_transform import (BYOLTransform,
                                               BYOLView1Transform,
                                               BYOLView2Transform)
from data_modules.dr_dataset import get_loo_dr

class DomainMapper():
    def __init__(self):
        self.unique_domains = [0,1,2,3]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x

class DRDM(pl.LightningDataModule):
    def __init__(self, cfg, leave_out: str) -> None:
        super().__init__()
        self.data_dir = cfg.data.path + '/DR'
        self.batch_size = cfg.param.batch_size

        if cfg.data.color_aug:
            self.train_transform = BYOLTransform(
            view_1_transform=T.Compose([
                BYOLView1Transform(
                    input_size=224, 
                    gaussian_blur=0.0,
                ),
            ]),
            view_2_transform=T.Compose([
                BYOLView2Transform(
                    input_size=224, 
                    gaussian_blur=0.0,
                ),
            ])
        )
        else:
            self.train_transform = BYOLTransform(
            view_1_transform=T.Compose([
                BYOLView1Transform(
                    input_size=224, 
                    gaussian_blur=0.0,
                    cj_prob=0.0,
                    random_gray_scale=0.0,
                    solarization_prob=0.0
                ),
            ]),
            view_2_transform=T.Compose([
                BYOLView2Transform(
                    input_size=224, 
                    gaussian_blur=0.0,
                    cj_prob=0.0,
                    random_gray_scale=0.0,
                    solarization_prob=0.0
                ),
            ])
        )

        self.val_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
        ])

        self.train_set, self.test_set = get_loo_dr(
            root=self.data_dir,
            leave_out=leave_out,
            train_tf=self.train_transform,
            test_tf=self.val_transform
        )

        train_set_knn, _ = get_loo_dr(
            root=self.data_dir,
            leave_out=leave_out,
            train_tf=self.val_transform,
            test_tf=self.val_transform
        )
        subset_size = 8_192
        range_tensor = torch.arange(len(train_set_knn))

        with torch.random.fork_rng():
            torch.manual_seed(42)
            indices = range_tensor[torch.randperm(len(range_tensor))[:subset_size]]
        
        self.train_set_knn = Subset(train_set_knn, indices)

        self.domain_mapper = DomainMapper()

        self.grouper = None

        self.cfg = cfg
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
            persistent_workers=True
        )
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:    
        train_loader_knn = DataLoader(
            self.train_set_knn,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        
        val_loader_knn = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        return [
            train_loader_knn,
            val_loader_knn
        ]
    
def main():
    pass

if __name__ == '__main__':
    main()