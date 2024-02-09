from typing import Any, Optional

import time
import hydra
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, optim
from torchmetrics import Accuracy
from torchvision import transforms as T
from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader, get_train_loader
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.wilds_dataset import WILDSSubset, WILDSDataset

import wandb
from models import DANN
from utils import DomainMapper

from tqdm import tqdm
import h5py

def get_full_dataloader(dataset: WILDSDataset, batch_size, transform) -> WILDSSubset:
    """Returns a dataloader for the complete dataset"""

    num_samples = len(dataset)
    subset = WILDSSubset(dataset, indices=list(range(num_samples)), transform=transform)

    return get_eval_loader("standard", subset, batch_size, num_workers=4)

class DPSmol(pl.LightningModule):
    def __init__(self, grouper, alpha, domain_mapper, weights, freeze, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.model = DANN(alpha=alpha, weights=weights)

        self.metric = Accuracy("binary")

        self.lr = kwargs["lr"]
        self.weight_decay = kwargs["weight_decay"]
        self.momentum = kwargs["momentum"]

        self.grouper: CombinatorialGrouper = grouper
        self.domain_mapper = domain_mapper
        self.freeze = list(freeze)

        self.alpha = alpha

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        X, t, M = batch
        d = self.domain_mapper(self.grouper.metadata_to_group(M.cpu())).to(self.device)

        loss_pred, loss_disc = self.model(X, t, d)

        self.log("loss_pred", loss_pred, prog_bar=True)
        self.log("loss_disc", loss_disc, prog_bar=True)

        if self.alpha == 0:
            loss_disc = 0.0

        return loss_pred + loss_disc
    
    def validation_step(self, batch, batch_idx, dataloader_idx) -> STEP_OUTPUT:
        if dataloader_idx == 0:
            loader_name = "val (ID)" 
        elif dataloader_idx == 1: 
            loader_name = "val (OOD)"
        else:
            loader_name = "test"

        X, t, M = batch 
        y, loss = self.model.forward_pred(X, t)

        acc = self.metric(y.argmax(dim=1), t)

        self.log(f"accuracy {loader_name}", acc, on_epoch=True)
        
        return loss 
    
    def configure_optimizers(self) -> Any:
        prms = []
        if 1 not in self.freeze:
            prms += (list(self.model.backbone.layer1.parameters()))
        if 2 not in self.freeze:
            prms += (list(self.model.backbone.layer2.parameters()))
        if 3 not in self.freeze:
            prms += (list(self.model.backbone.layer3.parameters()))
        if 4 not in self.freeze:
            prms += (list(self.model.backbone.layer4.parameters()))
        if 5 not in self.freeze:
            prms += (list(self.model.backbone.fc.parameters()))

        prms += (list(self.model.disc_head.parameters()))

        optimizer = optim.AdamW(
            prms, 
            lr=self.lr
        )

        return optimizer
    
    def compute_embeddings(self, name:str, train_loader: DataLoader, val_loader: DataLoader):
        hf = h5py.File(f'{name}-embeddings.h5', 'w')
        hf.create_dataset("train_embeddings", shape=(len(train_loader.dataset), 2048))
        hf.create_dataset("train_domains", shape=(len(train_loader.dataset)))
        hf.create_dataset("val_embeddings", shape=(len(val_loader.dataset), 2048))
        hf.create_dataset("val_domains", shape=(len(val_loader.dataset)))

        with torch.no_grad():

            for i, batch in enumerate(tqdm(train_loader)):
                x, t, m = batch
                bs = len(x)
                d = self.domain_mapper(self.grouper.metadata_to_group(m))

                y = self.model.embed(x.cuda()).squeeze()
                hf["train_embeddings"][i*32:i*32+bs] = y.cpu()
                hf["train_domains"][i*32:i*32+bs] = d

            for i, batch in enumerate(tqdm(val_loader)):
                x, t, m = batch
                bs = len(x)
                d = self.domain_mapper(self.grouper.metadata_to_group(m))

                y = self.model.embed(x.cuda()).squeeze()
                hf["val_embeddings"][i*32:i*32+bs] = y.cpu()
                hf["val_domains"][i*32:i*32+bs] = d

        hf.close()

    def compute_all_embeddings(self, name:str, dataloader: DataLoader, dm):
        hf = h5py.File(f'{name}-all-embeddings.h5', 'w')
        hf.create_dataset("embeddings", shape=(len(dataloader.dataset), 2048))
        hf.create_dataset("labels", shape=(len(dataloader.dataset)))
        hf.create_dataset("domains", shape=(len(dataloader.dataset)))

        with torch.no_grad():

            for i, batch in enumerate(tqdm(dataloader)):
                x, t, m = batch
                bs = len(x)
                d = dm(self.grouper.metadata_to_group(m))

                y = self.model.embed(x.cuda()).squeeze()

                hf["embeddings"][i*bs:(i+1)*bs] = y.cpu()
                hf["labels"][i*bs:(i+1)*bs] = t
                hf["domains"][i*bs:(i+1)*bs] = d

        hf.close()

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
            project="wilds-finetuning",
            
            # track hyperparameters and run metadata
            config={
                "learning_rate": cfg.param.lr,
                "weight_decay": cfg.param.weight_decay,
                "momentum": cfg.param.momentum,
                "batch_size": cfg.param.batch_size,

                "architecture": "ResNet 50",
                "dataset": "camelyon17",
            },

            id=f"{cfg.name}-{time.time()}"
        )
        logger = WandbLogger()

    ############################################################################
    torch.set_float32_matmul_precision('medium')

    dataset = get_dataset("camelyon17", root_dir=cfg.data_root)
    grouper = CombinatorialGrouper(dataset, ['hospital'])

    transform = T.Compose([
        T.ToTensor()
    ])

    train_set = dataset.get_subset("train", transform=transform)
    val_set_id = dataset.get_subset("id_val", transform=transform)
    val_set_ood = dataset.get_subset("val", transform=transform)
    test_set = dataset.get_subset("test", transform=transform)

    domain_mapper = DomainMapper(train_set.metadata_array[:,0])

    callback = ModelCheckpoint(
        monitor="accuracy val (ID)/dataloader_idx_0",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        accelerator="auto", 
        max_epochs=cfg.max_epochs, 
        logger=logger,
        callbacks=[callback]
    )

    model = DPSmol(
        lr=cfg.param.lr, 
        weight_decay=cfg.param.weight_decay, 
        momentum=cfg.param.momentum,
        grouper=grouper,
        alpha=cfg.disc.alpha,
        domain_mapper=domain_mapper,
        weights=cfg.weights,
        freeze=cfg.freeze
    )
    trainer.fit(
        model,
        train_dataloaders=get_train_loader("standard", train_set, batch_size=cfg.param.batch_size, num_workers=4),
        val_dataloaders=[
                get_eval_loader("standard", val_set_id, batch_size=cfg.param.batch_size, num_workers=4),
                get_eval_loader("standard", val_set_ood, batch_size=cfg.param.batch_size, num_workers=4),
                get_eval_loader("standard", test_set, batch_size=cfg.param.batch_size, num_workers=4)
        ]
    )

    model = DPSmol.load_from_checkpoint(callback.best_model_path)

    domain_mapper_2 = DomainMapper(dataset.metadata_array[:,0])

    model.compute_all_embeddings(
        name=f"{cfg.name}-{time.time()}",
        dataloader=get_full_dataloader(dataset, batch_size=cfg.param.batch_size, transform=transform),
        dm=domain_mapper_2
    )

if __name__ == "__main__":
    main()