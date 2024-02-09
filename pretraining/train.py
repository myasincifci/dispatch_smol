import hydra

from model import BarlowTwins

from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as L
import torch
import torch.nn as nn

from lightly.transforms.byol_transform import (BYOLTransform,
                                               BYOLView1Transform,
                                               BYOLView2Transform)
from lightly.transforms.utils import IMAGENET_NORMALIZE

from torchvision import transforms as T
from torchvision.models.resnet import ResNet50_Weights, resnet50
from wilds import get_dataset
from wilds.common.grouper import CombinatorialGrouper

from pytorch_lightning.loggers import WandbLogger
from utils import DomainMapper

import wandb


@hydra.main(version_base=None, config_path="configs")
def main(cfg: DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))

    torch.set_float32_matmul_precision("medium")

    logger = True
    if cfg.logging:
        # start a new wandb run to track this script
        wandb.login(
            key="deeed2a730495791be1a0158cf49240b65df1ffa"
        )
        wandb.init(
            # set the wandb project where this run will be logged
            project="barlow-twins-wilds",

            # track hyperparameters and run metadata
            config={
                "learning_rate": cfg.param.lr,
                "batch_size": cfg.param.batch_size,
                "architecture": "ResNet 50",
                "dataset": "camelyon17",
            }
        )
        logger = WandbLogger()

    L.seed_everything(42)

    # Transforms
    train_transform = BYOLTransform(
        view_1_transform=T.Compose([
            BYOLView1Transform(input_size=96, gaussian_blur=0.0),
        ]),
        view_2_transform=T.Compose([
            BYOLView2Transform(input_size=96, gaussian_blur=0.0),
        ])
    )

    val_transform = T.Compose([
        T.Resize((256, 256), antialias=True),
        T.ToTensor(),
        T.Normalize(
            mean=IMAGENET_NORMALIZE["mean"],
            std=IMAGENET_NORMALIZE["std"],
        ),
    ])

    # Datasets
    labeled_dataset = get_dataset(dataset=cfg.dataset.name, version=cfg.dataset.version,
                                  download=True, root_dir=cfg.data_path, unlabeled=False)

    train_set_labeled = labeled_dataset.get_subset(
        "train", transform=train_transform)

    if cfg.unlabeled:
        unlabeled_dataset = get_dataset(dataset=cfg.dataset.name,
                                        download=True, root_dir=cfg.data_path, unlabeled=True)
        train_set = unlabeled_dataset.get_subset(
            "train_unlabeled", transform=train_transform)
    else:
        train_set = train_set_labeled

    val_set = labeled_dataset.get_subset("val", transform=val_transform)

    train_set_knn = labeled_dataset.get_subset(
        "train", frac=4096/len(train_set_labeled), transform=val_transform)
    val_set_knn = labeled_dataset.get_subset(
        "val", frac=1024/len(val_set), transform=val_transform)

    grouper = CombinatorialGrouper(labeled_dataset, ['location'])

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.param.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4,
    )

    train_loader_knn = torch.utils.data.DataLoader(
        train_set_knn,
        batch_size=cfg.param.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4,
    )

    val_loader_knn = torch.utils.data.DataLoader(
        val_set_knn,
        batch_size=cfg.param.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )

    # Model
    backbone = resnet50(ResNet50_Weights.IMAGENET1K_V2)
    backbone.fc = nn.Identity()
    domain_mapper = DomainMapper(train_set_labeled.metadata_array[:, 0])

    barlow_twins = BarlowTwins(
        lr=cfg.param.lr, backbone=backbone,
        knn_loader=train_loader_knn,
        grouper=grouper,
        alpha=cfg.disc.alpha,
        domain_mapper=domain_mapper,
        cfg=cfg
    )

    trainer = L.Trainer(max_steps=25_000, accelerator="auto",
                        val_check_interval=100)
    trainer.fit(
        model=barlow_twins,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader_knn
    )

    # if args.save:
    #     torch.save(barlow_twins.backbone.state_dict(), args.save)


if __name__ == "__main__":
    main()
