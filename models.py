import torch
from torch import nn, optim
from torch.autograd import Function
from torchvision.models.resnet import resnet50


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class DANN(nn.Module):
    def __init__(self, alpha=0.8) -> None:
        super().__init__()
        self.backbone = self._make_backbone()
        self.pred_head = nn.Linear(2048, 2)
        self.disc_head = nn.Linear(2048, 3)

        self.crit_pred = nn.CrossEntropyLoss()
        self.crit_disc = nn.CrossEntropyLoss()

        self.alpha = alpha

    def _make_backbone(self) -> nn.Module:
        backbone = resnet50()
        backbone.fc = nn.Identity()

        return backbone
    
    def forward(self, x, t, d):
        x = self.backbone(x).squeeze()
        x_r = ReverseLayerF.apply(x, self.alpha)

        y = self.pred_head(x)
        z = self.disc_head(x_r)
        
        loss_pred = self.crit_pred(y, t)
        loss_disc = self.crit_disc(z, d)

        return loss_pred, loss_disc
    
    def forward_pred(self, x, t):
        x = self.backbone(x).squeeze()
        y = self.pred_head(x)
        loss = self.crit_pred(y, t)

        return y, loss
    
if __name__ == "__main__":
    model = DANN()