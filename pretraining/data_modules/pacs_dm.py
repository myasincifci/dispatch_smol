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
from data_modules.pacs_h5_dataset import get_pacs_loo
from utils import DomainMapper

class PacsDM(pl.LightningDataModule):
    def __init__(self, cfg, leave_out: List=None) -> None:
        super().__init__()
        self.data_dir = cfg.data_path
        self.batch_size = cfg.param.batch_size

        self.train_transform = BYOLTransform(
            view_1_transform=T.Compose([
                T.ToPILImage(),
                T.Resize(96),
                BYOLView1Transform(input_size=96, gaussian_blur=0.0),
            ]),
            view_2_transform=T.Compose([
                T.ToPILImage(),
                T.Resize(96),
                BYOLView2Transform(input_size=96, gaussian_blur=0.0),
            ])
        )

        self.val_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(96),
            T.ToTensor(),
            T.Normalize(
                mean=IMAGENET_NORMALIZE["mean"],
                std=IMAGENET_NORMALIZE["std"],
            ),
        ])

        self.train_set, self.test_set = get_pacs_loo(
            root=cfg.data_path,
            leave_out=leave_out,
            train_tf=self.train_transform,
            test_tf=self.val_transform
        )

        self.train_set_knn, self.test_set_knn = get_pacs_loo(
            root=cfg.data_path,
            leave_out=leave_out,
            train_tf=self.val_transform,
            test_tf=self.val_transform
        )

        self.grouper = None

        self.cfg = cfg
        self.domain_mapper = DomainMapper()
        self.num_classes = self.train_set.n_classes

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.domain_mapper = self.domain_mapper.setup(
                tensor(list(self.train_set.domains.values()))
            )
            
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
            # collate_fn=lambda x: x
        )
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:    
        train_loader_knn = DataLoader(
            self.train_set_knn,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
            pin_memory=True
        )

        val_loader_knn = DataLoader(
            self.test_set_knn,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
            pin_memory=True
        )

        return [
            train_loader_knn,
            val_loader_knn
        ]
    
def main():
    pass

if __name__ == '__main__':
    main()