import argparse

import pytorch_lightning as L
import torch
import torch.nn as nn
from data_modules.camelyon17_dm import CamelyonDM
from data_modules.pacs_dm import PacsDM
from data_modules.domainnet_dm import DomainNetDM
from model import BarlowTwins
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from resnet_mix import resnet50
from torchvision.models.resnet import ResNet50_Weights

import random
import wandb

def main(cfg_path: str) -> None:
    cfg = OmegaConf.load(cfg_path)

    while True:
        if '_BASE_' in cfg:
            base = cfg['_BASE_']
            del cfg['_BASE_']

            base_cfg = OmegaConf.load(base)
            cfg = OmegaConf.merge(base_cfg, cfg)

        else:
            break


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

    barlow_twins = BarlowTwins(
        num_classes=data_module.num_classes,
        backbone=backbone,
        grouper=data_module.grouper,
        domain_mapper=data_module.domain_mapper,
        cfg=cfg
    )
    barlow_twins = barlow_twins

    for param in barlow_twins.backbone.parameters():
        param.requires_grad = False

    trainer = L.Trainer(
        max_steps=cfg.trainer.max_steps,
        accelerator="auto",
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        logger=logger,
        log_every_n_steps=5,
    )

    trainer.fit(
        model=barlow_twins,
        datamodule=data_module,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--cfg-path', type=str, required=True, help='Path to the config file.')

    args = parser.parse_args()

    main(args.cfg_path)