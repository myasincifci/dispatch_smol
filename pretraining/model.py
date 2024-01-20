from typing import Any
from functools import partial

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torch.autograd import Function

import torch.optim as optim
from lightly.loss import BarlowTwinsLoss
from lightly.models.barlowtwins import BarlowTwinsProjectionHead

from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as L
from lightly.utils.benchmarking.knn import knn_predict

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
    def __init__(self, lr, backbone, knn_loader, grouper, alpha, domain_mapper, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(2048, cfg.param.projector_dim, cfg.param.projector_dim)

        if alpha > 0.0:
            self.crit_clf = nn.Linear(2048, 3)
            self.crit_crit = nn.CrossEntropyLoss()

        self.criterion = BarlowTwinsLoss()
        self.lr = lr
        self.dataloader_kNN = knn_loader

        self.num_classes = 2
        self.knn_k = 200
        self.knn_t = 0.1

        self.domain_mapper = domain_mapper
        self.grouper = grouper
        self.cfg = cfg

        self.alpha = alpha

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        if self.cfg.unlabeled:
            (x0, x1), metadata = batch
        else:
            (x0, x1), _, metadata = batch

        z0_, z1_ = self.backbone(x0).flatten(start_dim=1), self.backbone(x1).flatten(start_dim=1)
        z0, z1 = self.projection_head(z0_), self.projection_head(z1_)

        bt_loss = self.criterion(z0, z1)
        crit_loss = 0.0

        if self.alpha > 0.0:
            z = torch.cat([z0_, z1_], dim=0)
            group = self.grouper.metadata_to_group(metadata.cpu()).to(self.device)
            group = torch.cat([group, group], dim=0)
            group = self.domain_mapper(group)
            group = group.to(self.device)

            z = ReverseLayerF.apply(z, self.alpha)

            q = self.crit_clf(z)

            crit_loss = self.crit_crit(q, group)

            self.log("crit-loss", crit_loss.item(), prog_bar=True)

        self.log("bt-loss", bt_loss.item(), prog_bar=True)

        return bt_loss + crit_loss

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

        self._val_predicted_labels.clear()
        self._val_targets.clear()