import os
import torch
from torch.utils.data import random_split, Subset
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as TF
import pytorch_lightning as pl
from lightly.transforms.byol_transform import (BYOLTransform,
                                               BYOLView1Transform,
                                               BYOLView2Transform)
from lightly.transforms.utils import IMAGENET_NORMALIZE
from data_modules.domainnet_dataset import DomainNetDataset
from .domainnet_dataset import DomainNetDataset
from wilds.common.grouper import CombinatorialGrouper
from utils import DomainMapper

class SquarePadResize(torch.nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def forward(self, x):
        if type(x) is not torch.Tensor:
            x = TF.to_tensor(x)

        _, h, w = x.shape

        if h > w:
            padding = (h-w, 0)
        elif w > h:
            padding = (0, w-h)
        else:
            x = TF.resize(x, (self.size, self.size))
            return x


        x = TF.pad(x, padding)
        x = TF.resize(x, (self.size, self.size))

        return x

class DomainNetDM(pl.LightningDataModule):
    def __init__(self, cfg, unlabeled=False) -> None:
        super().__init__()

        self.cfg = cfg

        self.train_transform = BYOLTransform(
            view_1_transform=T.Compose([
                BYOLView1Transform(input_size=224, gaussian_blur=0.0),
            ]),
            view_2_transform=T.Compose([
                BYOLView2Transform(input_size=224, gaussian_blur=0.0),
            ])
        )

        self.val_transform = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(
                mean=IMAGENET_NORMALIZE["mean"],
                std=IMAGENET_NORMALIZE["std"],
            ),
        ])

        self.dataset = DomainNetDataset(
            root=os.path.join(self.cfg.data.path, 'domainnet_v1.0'), 
        )

        self.grouper = None # CombinatorialGrouper(self.dataset, ['domain'])

        self.domain_mapper = None# DomainMapper().setup(
        #     self.dataset.get_subset("train").metadata_array[:, 0]
        # )

        self.num_classes = self.dataset.n_classes

    def setup(self, stage: str) -> None:        
        if stage == 'fit':
            self.train_set, self.val_set = random_split(self.dataset, lengths=(0.9,0.1))

            self.knn_train_set = Subset(self.train_set, torch.randperm(len(self.train_set))[:60_000])
            
        elif stage == 'test':
            pass
        
        elif stage == 'predict':
            pass

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.cfg.param.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:   
        train_loader_knn = DataLoader(
            self.knn_train_set,
            batch_size=self.cfg.param.batch_size,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True
        )

        val_loader_id = DataLoader(
            self.val_set,
            batch_size=self.cfg.param.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True
        )

        return [
            train_loader_knn,
            val_loader_id,
        ]
    
def main():
    cfg = {
        'data': {
            'name': 'domainnet',
            'path': '/data',
            'num_workers': 0
        }
    } 
    cfg = DictConfig(cfg)

    dm = DomainNetDM(cfg)

    dm.setup(stage='fit')

    a=1

if __name__ == '__main__':
    from omegaconf import DictConfig

    main()