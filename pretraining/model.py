from functools import partial
from typing import Any
from types import MethodType

import pytorch_lightning as L
import torch
import torch.optim as optim
import torchmetrics
from lightly.loss import BarlowTwinsLoss
from lightly.models.barlowtwins import BarlowTwinsProjectionHead
from lightly.utils.benchmarking.knn import knn_predict
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
from torch.autograd import Function
from torch.nn import functional as F
from torch.distributions.beta import Beta
from torchvision.models.resnet import resnet50, ResNet50_Weights

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        if not self.training:
            return x
        
        if torch.rand(1).item() > self.p:
            return x

        B = x.size(0) # batch size

        mu = x.mean(dim=[2, 3], keepdim=True) # compute instance mean
        var = x.var(dim=[2, 3], keepdim=True) # compute instance variance
        sig = (var + self.eps).sqrt() # compute instance standard deviation
        mu, sig = mu.detach(), sig.detach() # block gradients
        x_normed = (x - mu) / sig # normalize input

        lmda = Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)).to(x.device) # sample instance-wise convex weights

        # if domain label is given:
        if False:
            # in this case, input x = [xˆi, xˆj]
            perm = torch.arange(B-1, -1, -1) # inverse index
            perm_j, perm_i = perm.chunk(2) # separate indices
            perm_j = perm_j[torch.randperm(B // 2)] # shuffling
            perm_i = perm_i[torch.randperm(B // 2)] # shuffling
            perm = torch.cat([perm_j, perm_i], 0) # concatenation
        else:
            perm = torch.randperm(B) # generate shuffling indices

        mu2, sig2 = mu[perm], sig[perm] # shuffling
        mu_mix = mu * lmda + mu2 * (1 - lmda) # generate mixed mean
        sig_mix = sig * lmda + sig2 * (1 - lmda) # generate mixed standard deviation

        return x_normed * sig_mix + mu_mix # denormalize input using the mixed statistics

def res50(cfg):
    if cfg.model.pretrained:
        model = resnet50(ResNet50_Weights.DEFAULT)
    else:
        model = resnet50()
    model.fc = nn.Linear(in_features=512, out_features=cfg.data.num_classes, bias=True)
    
    if cfg.mixstyle.active:
        model.ms = MixStyle(
            p=cfg.mixstyle.p,
            alpha=cfg.mixstyle.alpha,
            eps=cfg.mixstyle.eps
        )
    else:
        model.ms = None

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.ms(x) if self.ms else x

        x = self.layer2(x)
        x = self.ms(x) if self.ms else x

        x = self.layer3(x)
        x = self.ms(x) if self.ms else x

        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x 
    
    model.forward = MethodType(_forward, model)

    return model

class BarlowTwins(L.LightningModule):
    def __init__(self, num_classes, backbone, grouper, domain_mapper, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.backbone = backbone
        self.emb_dim = 2048
        self.projection_head = BarlowTwinsProjectionHead(
            self.emb_dim, cfg.model.projector_dim, cfg.model.projector_dim)

        if cfg.disc.alpha > 0.0:
            self.crit_clf = nn.Linear(self.emb_dim, len(domain_mapper.unique_domains))
            self.crit_crit = nn.CrossEntropyLoss()

        self.criterion = BarlowTwinsLoss()
        self.lr = cfg.param.lr

        self.num_classes = num_classes
        self.knn_k = 200
        self.knn_t = 0.1

        self.domain_mapper = domain_mapper
        self.grouper = grouper
        self.cfg = cfg

        self.BS = cfg.param.batch_size

        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        if self.cfg.unlabeled:
            (x0, x1), metadata = batch
        else:
            (x0, x1), _, metadata = batch

        z0_, z1_ = self.backbone(x0).flatten(
            start_dim=1), self.backbone(x1).flatten(start_dim=1)
        z0, z1 = self.projection_head(z0_), self.projection_head(z1_)

        bt_loss = self.criterion(z0, z1)
        crit_loss = 0.0

        if self.cfg.disc.alpha > 0.0:
            z = torch.cat([z0_, z1_], dim=0)

            if self.grouper:
                group = self.grouper.metadata_to_group(
                    metadata.cpu()).to(self.device)
                group = torch.cat([group, group], dim=0)
                group = self.domain_mapper(group)
                group = group.to(self.device)
            else:
                group = metadata

            z = ReverseLayerF.apply(z, self.cfg.disc.alpha)

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
                self._linear_warmup_decay(self.cfg.trainer.warmup),
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
        train = self.trainer.datamodule.train_dataloader()
        train_len = train.dataset.__len__()

        self.train_features = torch.zeros(
            (train_len, self.emb_dim), dtype=torch.float32, device=self.device)
        self.train_targets = torch.zeros(
            (train_len,), dtype=torch.float32, device=self.device)

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> None:
        bs = len(batch[0])

        if dataloader_idx == 0:  # knn-train
            X, t, _ = batch
            X = X.to(self.device)
            t = t.to(self.device)
            z = self.backbone(X).squeeze()
            z = F.normalize(z, dim=1)
            self.train_features[batch_idx *
                                self.BS:batch_idx*self.BS+bs] = z[:, :]
            self.train_targets[batch_idx*self.BS:batch_idx*self.BS+bs] = t[:]

        elif dataloader_idx > 0:  # knn-val
            X, t, _ = batch
            # torch.ones(self.BS, self.emb_dim).to(self.device)
            z = self.backbone(X).squeeze()
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

            self.accuracy(y[:, 0], t)
            self.log('val/accuracy', self.accuracy,
                     on_epoch=True, prog_bar=True)

    # def on_validation_epoch_end(self) -> None:
    #     # acc = self.accuracy.compute()
    #     # self.accuracy.reset()
    #     print(self.correct/1024)
