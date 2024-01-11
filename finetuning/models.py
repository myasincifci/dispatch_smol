import torch
from torch import nn, optim
from torch.autograd import Function
from torchvision.models.resnet import resnet50, ResNet50_Weights

from utils import get_backbone_from_ckpt


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
    def __init__(self, weights, alpha=0.8) -> None:
        super().__init__()
        self.backbone = self._make_backbone(weights)
        self.disc_head = nn.Linear(2048, 3)

        self.crit_pred = nn.CrossEntropyLoss()
        self.crit_disc = nn.CrossEntropyLoss()

        self.alpha = alpha
    
    def forward(self, x, t, d):
        for l in list(self.backbone.children())[:-1]:
            x = l(x)
        f = x.squeeze()
        f_r = ReverseLayerF.apply(f, self.alpha)

        y = self.backbone.fc(f)
        z = self.disc_head(f_r)
        
        loss_pred = self.crit_pred(y, t)
        loss_disc = self.crit_disc(z, d)

        return loss_pred, loss_disc
    
    def forward_pred(self, x, t):
        y = self.backbone(x)
        loss = self.crit_pred(y, t)

        return y, loss
    
    @torch.no_grad()
    def embed(self, x):
        for l in list(self.backbone.children())[:-1]:
            x = l(x)

        return x
    
    def _make_backbone(self, weights):
        if weights == "scratch":
            backbone = resnet50(num_classes=2)
        elif weights == "ImageNet":
            backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
            backbone.fc = nn.Linear(2048, 2)
        else:
            backbone = resnet50(num_classes=2)
            missing_keys, unexpected_keys = backbone.load_state_dict(get_backbone_from_ckpt(weights), strict=False)
            print("missing:", missing_keys, "unexpected:", unexpected_keys)

        return backbone
    
if __name__ == "__main__":
    model = DANN(weights="scratch")

    out = model.embed(torch.rand((32, 3, 96, 96)))
    print(out.shape)