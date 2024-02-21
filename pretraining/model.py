from typing import Any
from functools import partial

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torch.autograd import Function

import torch.optim as optim
from lightly.loss import BarlowTwinsLoss
from lightly.models.barlowtwins import BarlowTwinsProjectionHead
from lightly.utils.benchmarking.knn import knn_predict

from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as L
import torchmetrics

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
    def __init__(self, backbone, grouper, domain_mapper, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(2048, cfg.param.projector_dim, cfg.param.projector_dim)

        if cfg.disc.alpha > 0.0:
            self.crit_clf = nn.Linear(2048, len(domain_mapper.unique_domains))
            self.crit_crit = nn.CrossEntropyLoss()

        self.criterion = BarlowTwinsLoss()
        self.lr = cfg.param.lr

        self.num_classes = 2
        self.knn_k = 200
        self.knn_t = 0.1

        self.domain_mapper = domain_mapper
        self.grouper = grouper
        self.cfg = cfg

        self.BS = cfg.param.batch_size

        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=2)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        if self.cfg.unlabeled:
            (x0, x1), metadata = batch
        else:
            (x0, x1), _, metadata = batch

        z0_, z1_ = self.backbone(x0).flatten(start_dim=1), self.backbone(x1).flatten(start_dim=1)
        z0, z1 = self.projection_head(z0_), self.projection_head(z1_)

        bt_loss = self.criterion(z0, z1)
        crit_loss = 0.0

        if self.cfg.disc.alpha > 0.0:
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

        return bt_loss + self.cfg.disc.mult * crit_loss

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
        train, val = self.trainer.datamodule.val_dataloader()
        train_len = train.dataset.__len__()
        val_len = train.dataset.__len__()

        self.train_features = torch.zeros((train_len, 2048), dtype=torch.float32, device=self.device)
        self.train_targets = torch.zeros((train_len,), dtype=torch.float32, device=self.device)

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> None:
        bs = len(batch[0])    
        
        if dataloader_idx == 0: # knn-train
            X, t, _ = batch
            X = X.to(self.device)
            t = t.to(self.device)
            z = self.backbone(X).squeeze()
            z = F.normalize(z, dim=1)
            self.train_features[batch_idx*self.BS:batch_idx*self.BS+bs] = z[:,:]
            self.train_targets[batch_idx*self.BS:batch_idx*self.BS+bs] = t[:]

            # if self.train_features == []:
            #     self.train_features = z
            #     self.train_targets = t
            # else:
            #     self.train_features = torch.cat((self.train_features, z), dim=0).contiguous()
            #     self.train_targets = torch.cat((self.train_targets, t), dim=0).contiguous()

        elif dataloader_idx == 1: # knn-val
            X, t, _ = batch
            z = self.backbone(X).squeeze() # torch.ones(self.BS, 2048).to(self.device)
            z = F.normalize(z, dim=1)
            y = knn_predict(
                z,
                self.train_features.T,
                self.train_targets.to(torch.long),
                self.num_classes,
                self.knn_k,
                self.knn_t,
            )

            # self.correct += (y.argmax(dim=1) == t).to(torch.long).sum()

            self.accuracy(y[:,0], t)
            self.log('val/accuracy', self.accuracy, on_epoch=True, prog_bar=True)

    # def on_validation_epoch_end(self) -> None:
    #     # acc = self.accuracy.compute()
    #     # self.accuracy.reset()
    #     print(self.correct/1024)