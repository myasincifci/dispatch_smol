from typing import Any, Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, optim
from torchmetrics import Accuracy
from torchvision import transforms as T
from torchvision.models.densenet import densenet121
from torchvision.models.resnet import resnet18
from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader, get_train_loader
from wilds.common.grouper import CombinatorialGrouper

import wandb
from models import DANN
from utils import DomainMapper


class DPSmol(pl.LightningModule):
    def __init__(self, grouper, alpha, domain_mapper, *args: Any, **kwargs: Any) -> None:
        super().__init__()

        # self.model = densenet121(num_classes=2)
        # self.criterion = torch.nn.CrossEntropyLoss()
        self.model = DANN(alpha=alpha)

        self.metric = Accuracy("binary")

        self.lr = kwargs["lr"]
        self.weight_decay = kwargs["weight_decay"]
        self.momentum = kwargs["momentum"]

        self.grouper: CombinatorialGrouper = grouper
        self.domain_mapper = domain_mapper

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        X, t, M = batch
        d = self.domain_mapper(self.grouper.metadata_to_group(M.cpu())).cuda()

        loss_pred, loss_disc = self.model(X, t, d)

        self.log("loss_pred", loss_pred, prog_bar=True)
        self.log("loss_disc", loss_disc, prog_bar=True)

        return loss_pred + loss_disc
    
    def validation_step(self, batch, batch_idx, dataloader_idx) -> STEP_OUTPUT | None:
        loader_name = "ID" if dataloader_idx == 0 else "OOD"

        X, t, M = batch 
        y, loss = self.model.forward_pred(X, t)

        acc = self.metric(y.argmax(dim=1), t)

        self.log(f"accuracy ({loader_name})", acc, on_epoch=True)
        
        return loss 
    
    def configure_optimizers(self) -> Any:
        optimizer = optim.SGD(
            self.model.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay, 
            momentum=self.momentum
        )

        return optimizer

@hydra.main(version_base=None, config_path="configs")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    logger = True
    if cfg.logging:
        # start a new wandb run to track this script
        wandb.login(
            key="deeed2a730495791be1a0158cf49240b65df1ffa"
        )
        wandb.init(
            # set the wandb project where this run will be logged
            project="wild-finetuning",
            
            # track hyperparameters and run metadata
            config={
                "learning_rate": cfg.param.lr,
                "weight_decay": cfg.param.weight_decay,
                "momentum": cfg.param.momentum,
                "batch_size": cfg.param.batch_size,

                "architecture": "ResNet 50",
                "dataset": "camelyon17",
            }
        )
        logger = WandbLogger()

    ############################################################################
    torch.set_float32_matmul_precision('medium')

    dataset = get_dataset("camelyon17", root_dir="../data")
    grouper = CombinatorialGrouper(dataset, ['hospital'])

    transform = T.Compose([
        T.ToTensor()
    ])

    train_set = dataset.get_subset("train", transform=transform)
    val_set_id = dataset.get_subset("id_val", transform=transform)
    val_set_ood = dataset.get_subset("val", transform=transform)

    domain_mapper = DomainMapper(train_set.metadata_array[:,0])

    trainer = pl.Trainer(
        accelerator="auto", 
        max_epochs=cfg.max_epochs, 
        logger=logger,
    )

    trainer.fit(
        DPSmol(
            lr=cfg.param.lr, 
            weight_decay=cfg.param.weight_decay, 
            momentum=cfg.param.momentum,
            grouper=grouper,
            alpha=cfg.disc.alpha,
            domain_mapper=domain_mapper
        ),
        train_dataloaders=get_train_loader("standard", train_set, batch_size=cfg.param.batch_size, num_workers=4),
        val_dataloaders=[
                get_eval_loader("standard", val_set_id, batch_size=cfg.param.batch_size, num_workers=4),
                get_eval_loader("standard", val_set_ood, batch_size=cfg.param.batch_size, num_workers=4)
        ]
    )

if __name__ == "__main__":
    main()