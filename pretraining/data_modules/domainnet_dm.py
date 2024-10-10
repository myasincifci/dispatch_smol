import os
import json

import torch
from torch.utils.data import random_split, Subset
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF
import pytorch_lightning as pl
from lightly.transforms.byol_transform import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
)
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
            padding = (h - w, 0)
        elif w > h:
            padding = (0, w - h)
        else:
            x = TF.resize(x, (self.size, self.size))
            return x

        x = TF.pad(x, padding)
        x = TF.resize(x, (self.size, self.size))

        return x


class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        batch = self.dataset[index]
        if self.transform:
            x = self.transform(batch["image"])
        else:
            x = batch["image"]
        y = batch["label"]
        d = batch["domain"]

        return dict(image=x, label=y, domain=d)

    def __len__(self):
        return len(self.dataset)


class DomainNetDM(pl.LightningDataModule):
    def __init__(self, cfg, unlabeled=False) -> None:
        super().__init__()

        self.cfg = cfg

        self.train_transform = BYOLTransform(
            view_1_transform=T.Compose(
                [
                    BYOLView1Transform(input_size=cfg.data.img_size, gaussian_blur=0.0),
                ]
            ),
            view_2_transform=T.Compose(
                [
                    BYOLView2Transform(input_size=cfg.data.img_size, gaussian_blur=0.0),
                ]
            ),
        )

        self.val_transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize((cfg.data.img_size, cfg.data.img_size)),
                T.Normalize(
                    mean=IMAGENET_NORMALIZE["mean"],
                    std=IMAGENET_NORMALIZE["std"],
                ),
            ]
        )

        with open('data_modules/train_set_map_domainnet.json', 'r') as file:
            train_set_map = json.load(file)

        with open('data_modules/test_set_map_domainnet.json', 'r') as file:
            test_set_map = json.load(file)

        self.train_set_ = DomainNetDataset(train_set_map)
        self.val_set_ = DomainNetDataset(test_set_map)

        # self.dataset = DomainNetDataset(
        #     root=os.path.join(self.cfg.data.path, "domainnet_v1.0"),
        # )

        self.grouper = None
        self.domain_mapper = None

        self.num_classes = self.train_set_.num_classes

    def setup(self, stage: str) -> None:
        if stage == "fit":
            # train_set, val_set = random_split(self.dataset, lengths=(0.9, 0.1))
            self.train_set = TransformDataset(self.train_set_, self.train_transform)
            self.val_set = TransformDataset(self.val_set_, self.val_transform)

            knn_train_set = Subset(
                self.train_set_, torch.randperm(len(self.train_set_))[:60_000]
            )
            self.knn_train_set = TransformDataset(knn_train_set, self.val_transform)

        elif stage == "test":
            pass

        elif stage == "predict":
            pass

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.cfg.param.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        train_loader_knn = DataLoader(
            self.knn_train_set,
            batch_size=self.cfg.param.batch_size,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        val_loader_id = DataLoader(
            self.val_set,
            batch_size=self.cfg.param.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        return [
            train_loader_knn,
            val_loader_id,
        ]


def main():
    cfg = {"data": {"name": "domainnet", "path": "/data", "num_workers": 0}}
    cfg = DictConfig(cfg)

    dm = DomainNetDM(cfg)

    dm.setup(stage="fit")

    a = 1


if __name__ == "__main__":
    from omegaconf import DictConfig

    main()
