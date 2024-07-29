import os
import random

import hydra
import pytorch_lightning as L
import torch
import torch.nn as nn
from data_modules.camelyon17_dm import CamelyonDM
# from pretraining.data_modules.pacs_h5_dm import PacsDM
from data_modules.pacs_dm import PacsDM
from data_modules.rxrx1_dm import RxRx1DM
from model import BarlowTwins
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision import transforms as T
from torchvision.models.resnet import ResNet50_Weights, ResNet18_Weights
from resnet_mix import resnet18, resnet50

import wandb


@hydra.main(version_base=None, config_path="configs")
def main(cfg: DictConfig) -> None:
    print(os.getcwd())
    print(OmegaConf.to_yaml(cfg))

    torch.set_float32_matmul_precision("medium")

    logger = True
    if cfg.logging:
        wandb.init(
            project=cfg.logger.project,
            config=OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )
        )
        logger = WandbLogger()

    seed = random.randint(0,9999999)
    L.seed_everything(seed, workers=True)
    print(f'Seed: {seed}')

    # Data
    # data_module = CamelyonDM(cfg)
    # data_module = RxRx1DM(cfg)
    data_module = PacsDM(cfg)

    # Model
    if cfg.model.pretrained:
        backbone = resnet50(
            ResNet50_Weights.DEFAULT,
            p=cfg.mixstyle.p,
            alpha=cfg.mixstyle.alpha,
            eps=cfg.mixstyle.eps,
        )
    else:
        backbone = resnet50(
            p=cfg.mixstyle.p,
            alpha=cfg.mixstyle.alpha,
            eps=cfg.mixstyle.eps,
        )
    backbone.fc = nn.Identity()

    barlow_twins = BarlowTwins(
        num_classes=data_module.num_classes,
        backbone=backbone,
        grouper=data_module.grouper,
        domain_mapper=data_module.domain_mapper,
        cfg=cfg,
        dm=data_module
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = L.Trainer(
        max_steps=cfg.trainer.max_steps,
        accelerator="auto",
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        logger=logger,
        log_every_n_steps=5,
        callbacks=[lr_monitor]
    )

    trainer.fit(
        model=barlow_twins,
        datamodule=data_module
    )


if __name__ == "__main__":
    main()
