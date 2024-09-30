import os
import random

import hydra
import pytorch_lightning as L
import torch
import torch.nn as nn
from data_modules.camelyon17_dm import CamelyonDM
from data_modules.pacs_dm import PacsDM
from data_modules.domainnet_dm import DomainNetDM

from model import BarlowTwins, HeadPretrain
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision import transforms as T
from torchvision.models.resnet import ResNet50_Weights, ResNet18_Weights
from resnet_mix import resnet18, resnet50

import random
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
    print(f'Seed:', seed)

    # Data TODO: do properly
    match cfg.data.name:
        case 'camelyon':
            data_module = CamelyonDM(cfg)
        case 'pacs':
            data_module = PacsDM(cfg, leave_out=['sketch'])
        case 'domainnet':
            data_module = DomainNetDM(cfg)
        case _:
            raise Exception('Invalid Dataset')

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

    if cfg.head_pretrain.active:
        model = HeadPretrain(
            backbone, cfg
        )

        trainer = L.Trainer(
            max_epochs=cfg.head_pretrain.epochs,
            logger=logger,
        )

        trainer.fit(
            model=model,
            datamodule=data_module
        )

        head_weights = model.projection_head.state_dict()
    else:
        head_weights = None

    barlow_twins = BarlowTwins(
        num_classes=data_module.num_classes,
        backbone=backbone,
        head_weights=head_weights,
        grouper=data_module.grouper,
        domain_mapper=data_module.domain_mapper,
        cfg=cfg,
        dm=data_module
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator="auto",
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        logger=logger,
        callbacks=[lr_monitor],
        num_sanity_val_steps=-1,
    )

    trainer.fit(
        model=barlow_twins,
        datamodule=data_module
    )


if __name__ == "__main__":
    main()
