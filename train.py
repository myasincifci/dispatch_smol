from typing import Any, Optional

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, optim
from torchmetrics import Accuracy
from torchvision import transforms as T
from torchvision.models.resnet import resnet18
from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader, get_train_loader
from wilds.common.grouper import CombinatorialGrouper

from utils import DomainMapper


class DPSmol(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.model = resnet18(num_classes=2)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.metric = Accuracy("binary")

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        X, t, M = batch
        y = self.model(X)

        loss = self.criterion(y,t)

        self.log("loss", loss, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        X, t, M = batch 
        y = self.model(X)

        loss = self.criterion(y, t) 
        acc = self.metric(y, t)

        self.log('accuracy', acc, on_epoch=True)
        
        return loss 
    
    def configure_optimizers(self) -> Any:
        optimizer = optim.SGD(self.model.parameters(), lr=1e-3)

        return optimizer

def main():
     # start a new wandb run to track this script
    wandb.login(
        key="deeed2a730495791be1a0158cf49240b65df1ffa"
    )
    wandb.init(
        # set the wandb project where this run will be logged
        project="wild-finetuning",
        
        # track hyperparameters and run metadata
        config={
            "learning_rate": 1e-3,
            "batch_size": 64,
            "architecture": "ResNet 50",
            "dataset": "camelyon17",
        }
    )

    ############################################################################

    dataset = get_dataset("camelyon17", root_dir="../data")
    domain_mapper = DomainMapper(dataset.metadata_array[:,0])
    grouper = CombinatorialGrouper(dataset, ['hospital'])

    transform = T.Compose([
        T.ToTensor()
    ])

    train_set = dataset.get_subset("train", transform=transform)
    val_set_id = dataset.get_subset("id_val", transform=transform)

    wandb_logger = WandbLogger()

    trainer = pl.Trainer(accelerator="auto", max_epochs=50, logger=wandb_logger)

    trainer.fit(
        DPSmol(),
        get_train_loader("standard", train_set, batch_size=64, num_workers=4),
        get_eval_loader("standard", val_set_id, batch_size=64, num_workers=4)
    )

if __name__ == "__main__":
    main()