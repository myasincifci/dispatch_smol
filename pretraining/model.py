from functools import partial
from typing import Any, Mapping

import pytorch_lightning as L
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import torchmetrics
from lightly.loss import BarlowTwinsLoss
from lightly.models.barlowtwins import BarlowTwinsProjectionHead
from lightly.utils.benchmarking.knn import knn_predict
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
from torch.autograd import Function
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression

class HeadPretrain(L.LightningModule):
    def __init__(self, backbone, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.backbone: nn.Module = backbone
        self.emb_dim = [m for m in backbone.modules() if isinstance(m, torch.nn.Conv2d)][-1].out_channels

        self.projection_head = BarlowTwinsProjectionHead(
            self.emb_dim, cfg.model.projector_dim, cfg.model.projector_dim)

        # Disable gradients for backbone parameters
        self.backbone.requires_grad_(False)

        # Disable batch-norm running mean/var
        self.backbone.eval()
        # for _, module in self.backbone.named_modules():
        #     if isinstance(module, torch.nn.BatchNorm2d):
        #         module.eval()

        self.criterion = BarlowTwinsLoss()

        self.cfg = cfg

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        (x0, x1) = batch["image"]

        z0_, z1_ = self.backbone(x0).flatten(
            start_dim=1), self.backbone(x1).flatten(start_dim=1)
        z0, z1 = self.projection_head(z0_), self.projection_head(z1_)

        bt_loss = self.criterion(z0, z1)

        self.log("head/bt-loss", bt_loss.item(), prog_bar=True)

        return bt_loss

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(
            params=self.parameters(), 
            lr=self.cfg.head_pretrain.lr,
        )

        return [optimizer]

class BarlowTwins(L.LightningModule):
    def __init__(self, num_classes, backbone, head_weights, grouper, domain_mapper, cfg, dm, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.backbone = backbone
        self.emb_dim = 2048
        self.projection_head = BarlowTwinsProjectionHead(
            self.emb_dim, cfg.model.projector_dim, cfg.model.projector_dim)
        
        if cfg.head_pretrain.active:
            self.projection_head.load_state_dict(head_weights)

            self.requires_grad_(True)
            self.backbone.eval()

        self.criterion = BarlowTwinsLoss()
        self.lr = cfg.param.lr

        self.num_classes = num_classes
        self.knn_k = 200
        self.knn_t = 0.1

        self.domain_mapper = domain_mapper
        self.grouper = grouper
        self.cfg = cfg

        self.BS = cfg.param.batch_size
        self.dm = dm

        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes)
        
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        (x0, x1) = batch["image"]

        z0_, z1_ = self.backbone(x0).flatten(
            start_dim=1), self.backbone(x1).flatten(start_dim=1)
        z0, z1 = self.projection_head(z0_), self.projection_head(z1_)

        bt_loss = self.criterion(z0, z1)

        self.log("bt-loss", bt_loss.item(), prog_bar=True)

        return bt_loss

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(params=self.parameters(), lr=self.lr,)

        scheduler = {
            "scheduler": self.get_linear_warmup_cos_annealing(
                optimizer=optimizer,
                warmup_iters=10*(len(self.dm.train_set)//self.cfg.param.batch_size),
                # total_iters=self.cfg.trainer.max_steps
                total_iters=self.cfg.trainer.max_epochs*(len(self.dm.train_set)//self.cfg.param.batch_size)
            ),
            "interval": "step",
            "frequency": 1
        }

        return [optimizer], [scheduler]

    def _fn(self, warmup_steps, step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            return 1.0

    def _linear_warmup_decay(self, warmup_steps):
        return partial(self._fn, warmup_steps)
    
    def get_linear_warmup_cos_annealing(self, optimizer, warmup_iters, total_iters):
        scheduler_warmup = LinearLR(optimizer, total_iters=warmup_iters, start_factor=1e-100)
        scheduler_cos_decay = CosineAnnealingLR(optimizer, T_max=total_iters-warmup_iters, eta_min=self.cfg.param.lr/1000)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, 
                                    scheduler_cos_decay], milestones=[warmup_iters])

        return scheduler

    def on_validation_epoch_start(self) -> None:
        train, val, *_ = self.trainer.datamodule.val_dataloader()
        train_len = train.dataset.__len__()
        val_len = train.dataset.__len__()

        self.train_features = torch.zeros(
            (train_len, self.emb_dim), dtype=torch.float32, device=self.device)
        self.train_targets = torch.zeros(
            (train_len,), dtype=torch.float32, device=self.device)
        
        self.clf = None

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if dataloader_idx == 1 and batch_idx == 0:
            print('Training Logistic Regression...')
            self.clf = LogisticRegression(random_state=42).fit(self.train_features.detach().cpu(), self.train_targets.detach().cpu())


    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> None:
        bs = len(batch['image'])

        if dataloader_idx == 0:  # embed train features
            X, t = batch['image'], batch['label']
            X = X.to(self.device)
            t = t.to(self.device)
            z = self.backbone(X).squeeze()
            z = F.normalize(z, dim=1)
            self.train_features[batch_idx *
                                self.BS:batch_idx*self.BS+bs] = z[:, :]
            self.train_targets[batch_idx*self.BS:batch_idx*self.BS+bs] = t[:]

        elif dataloader_idx > 0:  # validate
            X, t = batch['image'], batch['label']

            # torch.ones(self.BS, self.emb_dim).to(self.device)
            z = self.backbone(X).squeeze()
            z = F.normalize(z, dim=1)
            # y = knn_predict(
            #     z,
            #     self.train_features.T,
            #     self.train_targets.to(torch.long),
            #     self.num_classes,
            #     self.knn_k,
            #     self.knn_t,
            # )

            y = self.clf.predict(z.detach().cpu())
            y = torch.from_numpy(y).to(self.device)

            self.accuracy(y, t)
            self.log('val/accuracy', self.accuracy,
                     on_epoch=True, prog_bar=True)
