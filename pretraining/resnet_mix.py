from typing import Any, Type, Union, List, Optional, Callable

import torch
from torch import nn, Tensor
from torchvision.models.resnet import Bottleneck, ResNet, BasicBlock
from torch.distributions.beta import Beta

from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param

class MixStyle(nn.Module):
    def __init__(self, p, alpha, eps, *args, **kwargs) -> None:
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


class ResNetMix(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        p=1.0,
        alpha=0.1,
        eps=1e-6
    ) -> None:
        super().__init__(
            block,
            layers,
            num_classes,
            zero_init_residual,
            groups,
            width_per_group,
            replace_stride_with_dilation,
            norm_layer
        )
        print(f'p: {p}')
        print(f'alpha: {alpha}')
        print(f'eps: {eps}')

        self.mix = MixStyle(p, alpha, eps)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.mix(x)

        x = self.layer2(x)
        x = self.mix(x)

        x = self.layer3(x)
        x = self.mix(x)

        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNetMix(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model

def resnet18(weights=None, num_classes=1000, **kwargs):    
    return _resnet(
        BasicBlock,
        [2, 2, 2, 2],
        weights=weights,
        progress=True,
        norm_layer=nn.BatchNorm2d,
        num_classes=num_classes,
        **kwargs
    )

def resnet50(weights=None, num_classes=1000, **kwargs):    
    return _resnet(
        Bottleneck,
        [3, 4, 6, 3],
        weights=weights,
        progress=True,
        norm_layer=nn.BatchNorm2d,
        num_classes=num_classes,
        **kwargs
    )