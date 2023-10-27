from typing import Any, Optional
from torchmetrics import Accuracy

from utils import DomainMapper

import torch
from torch import optim, nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchvision.models.resnet import resnet18
from torchvision import transforms as T

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper

class DPSmol(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.model = resnet18(num_classes=2)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.metric = Accuracy("binary")

    def training_step(self, batch) -> STEP_OUTPUT:
        X, t, M = batch
        y = self.model(X)

        loss = self.criterion(y,t)

        self.log("loss", loss, prog_bar=True)

        return loss
    
    def validation_step(self, batch) -> STEP_OUTPUT | None:
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
    dataset = get_dataset("camelyon17", root_dir="../data")
    domain_mapper = DomainMapper(dataset.metadata_array[:,0])
    grouper = CombinatorialGrouper(dataset, ['hospital'])

    transform = T.Compose([
        T.ToTensor()
    ])

    train_set = dataset.get_subset("train", transform=transform)
    val_set_id = dataset.get_subset("id_val", transform=transform)

    trainer = pl.Trainer(accelerator="auto", max_epochs=1)

    trainer.fit(
        DPSmol(),
        get_train_loader("standard", train_set, batch_size=64)
    )

if __name__ == "__main__":
    main()