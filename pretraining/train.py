import hydra

from model import BarlowTwins
from data_modules.camelyon17_dm import CamelyonDM

from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as L
import torch
import torch.nn as nn

from torchvision import transforms as T
from torchvision.models.resnet import ResNet50_Weights, resnet50

from pytorch_lightning.loggers import WandbLogger

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

    # Data
    data_module = CamelyonDM(cfg)

    # Model
    backbone = resnet50(ResNet50_Weights.IMAGENET1K_V2)
    backbone.fc = nn.Identity()

    barlow_twins = BarlowTwins(
        backbone=backbone,
        grouper=data_module.grouper,
        domain_mapper=data_module.domain_mapper,
        cfg=cfg
    )

    trainer = L.Trainer(max_steps=25_000, accelerator="auto",
                        val_check_interval=100)
    trainer.fit(
        model=barlow_twins,
        datamodule=data_module
    )

    # if args.save:
    #     torch.save(barlow_twins.backbone.state_dict(), args.save)


if __name__ == "__main__":
    main()
