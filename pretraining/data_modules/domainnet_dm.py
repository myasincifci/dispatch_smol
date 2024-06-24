import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as TF
import pytorch_lightning as pl
from lightly.transforms.byol_transform import (BYOLTransform,
                                               BYOLView1Transform,
                                               BYOLView2Transform)
from lightly.transforms.utils import IMAGENET_NORMALIZE
from domainnet_dataset import DomainNetDataset
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
                BYOLView1Transform(input_size=200, gaussian_blur=0.0), # TODO: adjust input size
            ]),
            view_2_transform=T.Compose([
                BYOLView2Transform(input_size=200, gaussian_blur=0.0), # TODO: adjust input size
            ])
        )

        self.val_transform = T.Compose([
            T.ToTensor(),
            SquarePadResize(200),
            T.Normalize(
                mean=IMAGENET_NORMALIZE["mean"],
                std=IMAGENET_NORMALIZE["std"],
            ),
        ])

    def setup(self, stage: str) -> None:
        self.dataset = DomainNetDataset(
            download=False, 
            root_dir=self.cfg.data.path, 
        )

        self.grouper = CombinatorialGrouper(self.dataset, ['domain']) # TODO: fix name of domain

        self.domain_mapper = DomainMapper().setup(
            self.dataset.get_subset("train").metadata_array[:, 0]
        )

        self.num_classes = self.dataset.n_classes

        
        if stage == 'fit':
            self.train_set = self.dataset.get_subset(
                    "train", 
                    transform=self.train_transform
            )

            self.val_set_id = self.dataset.get_subset(
                "id_test", 
                transform=self.val_transform
            )

            self.val_set_ood = self.dataset.get_subset(
                "val", 
                transform=self.val_transform
            )

            self.test_set_ood = self.dataset.get_subset(
                "test",
                transform=self.val_transform
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
            num_workers=self.cfg.data.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:    
        val_loader_id = DataLoader(
            self.val_set_knn_id,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True
        )

        val_loader_ood = DataLoader(
            self.val_set_knn,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True
        )

        test_loader_ood = DataLoader(
            self.test_set_knn,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True
        )

        return [
            val_loader_id,
            val_loader_ood,
            test_loader_ood
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