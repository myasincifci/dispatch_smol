from argparse import Namespace
from functools import partial
from typing import Any, Sequence, Tuple, Union

import hydra

from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightly.loss import BarlowTwinsLoss
from lightly.models.barlowtwins import BarlowTwinsProjectionHead
from lightly.transforms.byol_transform import (BYOLTransform,
                                               BYOLView1Transform,
                                               BYOLView2Transform)
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking.knn import knn_predict
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.autograd import Function
from torchmetrics.functional import accuracy
from torchvision import transforms as T
from torchvision.models.resnet import ResNet50_Weights, resnet50
from wilds import get_dataset
from wilds.common.grouper import CombinatorialGrouper

from pytorch_lightning.loggers import WandbLogger
from utils import DomainMapper

import wandb

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class BarlowTwins(L.LightningModule):
    def __init__(self, lr, backbone, knn_loader, grouper, alpha, domain_mapper, cfg, dom_crit=True, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(2048, cfg.param.projector_dim, cfg.param.projector_dim)

        if dom_crit:
            self.crit_clf = nn.Linear(2048, 3)
            self.crit_crit = nn.NLLLoss()

        self.criterion = BarlowTwinsLoss()
        self.lr = lr
        self.dataloader_kNN = knn_loader

        self.num_classes = 2
        self.knn_k = 200
        self.knn_t = 0.1

        self.dom_crit = dom_crit
        self.domain_mapper = domain_mapper
        self.grouper = grouper
        self.cfg = cfg

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        if self.cfg.unlabeled:
            (x0, x1), metadata = batch
        else:
            (x0, x1), t, metadata = batch
        bs = x0.shape[0]

        z0, z1 = self.backbone(x0).view(bs, -1), self.backbone(x1).view(bs, -1)
        z0, z1 = self.projection_head(z0), self.projection_head(z1)

        clf_loss = self.criterion(z0, z1)
        crit_loss = 0.0
        if self.dom_crit:
            z = torch.cat([z0, z1])
            group = self.grouper.metadata_to_group(metadata.cpu()).to(self.device)
            group = torch.cat([group, group])
            group = self.domain_mapper(group)

            z = ReverseLayerF.apply(z, 1.0)

            q = self.crit_clf(z)

            crit_loss = self.crit_crit(q, group)

            # wandb.log({"crit-loss": crit_loss.item()})
            self.log("crit-loss", crit_loss.item(), prog_bar=True)

        # wandb.log({"bt-loss": clf_loss.item()})
        self.log("bt-loss", clf_loss.item(), prog_bar=True)

        return clf_loss + crit_loss*15.0

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(params=self.parameters(), lr=self.lr)

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                self._linear_warmup_decay(1000),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def _fn(self, warmup_steps, step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            return 1.0

    def _linear_warmup_decay(self, warmup_steps):
        return partial(self._fn, warmup_steps)
    
    def on_validation_epoch_start(self) -> None:
        self._val_predicted_labels = []
        self._val_targets = []
        self.max_accuracy = 0.0

        train_features = []
        train_targets = []
        with torch.no_grad():
            for data in self.dataloader_kNN:
                img, target, _ = data
                img = img.to(self.device)
                target = target.to(self.device)
                feature = self.backbone(img).squeeze()
                feature = F.normalize(feature, dim=1)
                train_features.append(feature)
                train_targets.append(target)
        self._train_features = torch.cat(train_features, dim=0).t().contiguous()
        self._train_targets = torch.cat(train_targets, dim=0).t().contiguous()
    
    def validation_step(self, batch, batch_idx) -> None:
        # we can only do kNN predictions once we have a feature bank
        if self._train_features is not None and self._train_targets is not None:
            images, targets, _ = batch
            feature = self.backbone(images).squeeze()
            feature = F.normalize(feature, dim=1)
            predicted_labels = knn_predict(
                feature,
                self._train_features,
                self._train_targets,
                self.num_classes,
                self.knn_k,
                self.knn_t,
            )

            self._val_predicted_labels.append(predicted_labels.cpu())
            self._val_targets.append(targets.cpu())

    def on_validation_epoch_end(self) -> None:
        if self._val_predicted_labels and self._val_targets:
            predicted_labels = torch.cat(self._val_predicted_labels, dim=0)
            targets = torch.cat(self._val_targets, dim=0)
            top1 = (predicted_labels[:, 0] == targets).float().sum()
            acc = top1 / len(targets)
            if acc > self.max_accuracy:
                self.max_accuracy = acc.item()
            self.log("kNN_accuracy", acc * 100.0, prog_bar=True)
            # wandb.log({"kNN_accuracy": acc * 100.0})

        self._val_predicted_labels.clear()
        self._val_targets.clear()

@hydra.main(version_base=None, config_path="configs")
def main(cfg : DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))
    print(f"Using GPU: {torch.cuda.is_available()}")

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
                "dataset": cfg.dataset,
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
        T.ToTensor(),
        T.Normalize(
            mean=IMAGENET_NORMALIZE["mean"],
            std=IMAGENET_NORMALIZE["std"],
        ),
    ])


    # Datasets
    labeled_dataset = get_dataset(dataset=cfg.dataset.name, version=str(cfg.dataset.version),
                          download=True, root_dir=cfg.data_path, unlabeled=False)   

    train_set_labeled = labeled_dataset.get_subset("train", transform=train_transform)

    if cfg.unlabeled:
        unlabeled_dataset = get_dataset(dataset=cfg.dataset,
                          download=True, root_dir=cfg.data_path, unlabeled=True) 
        train_set = unlabeled_dataset.get_subset("train_unlabeled", transform=train_transform)
    else:
        train_set = train_set_labeled

    val_set = labeled_dataset.get_subset("val", transform=val_transform)


    train_set_knn = labeled_dataset.get_subset(
        "train", frac=4096/len(train_set_labeled), transform=val_transform)
    val_set_knn = labeled_dataset.get_subset(
        "val", frac=1024/len(val_set), transform=val_transform)
    
    grouper = CombinatorialGrouper(labeled_dataset, ['hospital'])

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
        shuffle=True,
        drop_last=False,
        num_workers=4,
    )

    # Model
    backbone = resnet50(ResNet50_Weights.IMAGENET1K_V2)
    backbone.fc = nn.Identity()
    domain_mapper = DomainMapper(train_set_labeled.metadata_array[:,0])

    barlow_twins = BarlowTwins(
        lr=cfg.param.lr, backbone=backbone, 
        knn_loader=train_loader_knn, 
        grouper=grouper,
        alpha=cfg.disc.alpha,
        dom_crit=False,
        domain_mapper=domain_mapper,
        cfg=cfg
    )

    trainer = L.Trainer(
        max_steps=25_000, 
        accelerator="auto", 
        val_check_interval=100,
        logger=logger
    )
    trainer.fit(
        model=barlow_twins, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader_knn
    )

    # if args.save:
    #     torch.save(barlow_twins.backbone.state_dict(), args.save)

if __name__ == "__main__":
    main()