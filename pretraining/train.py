import hydra

from model import BarlowTwins
from data_modules.camelyon17_dm import CamelyonDM
from data_modules.rxrx1_dm import RxRx1DM
from data_modules.pacs_dm import PacsDM

from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as L
import torch
import torch.nn as nn

from torchvision import transforms as T
from torchvision.models.resnet import ResNet50_Weights, resnet50

from pytorch_lightning.loggers import WandbLogger

import wandb

import os

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
            project="dispatch-pretrain-pacs",
            config=None #cfg
        )
        logger = WandbLogger()

    L.seed_everything(42, workers=True)

    # Data
    data_module = CamelyonDM(cfg)
    # data_module = RxRx1DM(cfg)
    # data_module = PacsDM(cfg)

    # Model
    backbone = resnet50()#ResNet50_Weights.IMAGENET1K_V2)
    backbone.fc = nn.Identity()

    barlow_twins = BarlowTwins(
        num_classes=data_module.num_classes,
        backbone=backbone,
        grouper=data_module.grouper,
        domain_mapper=data_module.domain_mapper,
        cfg=cfg
    )

    trainer = L.Trainer(
        max_steps=50_000, 
        accelerator="auto",
        check_val_every_n_epoch=1,
        logger=logger,
    )

    trainer.fit(
        model=barlow_twins,
        datamodule=data_module
    )

if __name__ == "__main__":
    main()
