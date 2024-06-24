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
from wilds import get_dataset
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
        self.dataset = get_dataset(
            dataset=self.cfg.data.name,
            download=True, 
            root_dir=self.data_dir, 
            unlabeled=False
        )

        self.grouper = CombinatorialGrouper(self.dataset, ['domain']) # TODO: fix name of domain

        self.domain_mapper = DomainMapper().setup(
            self.labeled_dataset.get_subset("train").metadata_array[:, 0]
        )

        self.num_classes = self.labeled_dataset.n_classes

        
        if stage == 'fit':
            train_set = self.dataset.get_subset(
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

            ################

            self.train_set_knn = self.labeled_dataset.get_subset(
                "train", 
                frac=4096/len(train_set_labeled), 
                transform=self.val_transform
            )

            self.val_set_knn_id = self.labeled_dataset.get_subset(
                "id_test",
                frac=2048/len(self.val_set_id), 
                transform=self.val_transform
            )

            self.val_set_knn = self.labeled_dataset.get_subset(
                "val", 
                frac=2048/len(self.val_set), 
                transform=self.val_transform
            )

            self.test_set_knn = self.labeled_dataset.get_subset(
                "test",
                frac=2048/len(self.test_set),
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
        train_loader_knn = DataLoader(
            self.train_set_knn,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True
        )

        val_loader_knn_id = DataLoader(
            self.val_set_knn_id,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True
        )

        val_loader_knn = DataLoader(
            self.val_set_knn,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True
        )

        test_loader_knn = DataLoader(
            self.test_set_knn,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True
        )

        return [
            train_loader_knn,
            val_loader_knn_id,
            val_loader_knn,
            test_loader_knn
        ]