from typing import OrderedDict, List, Dict

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

def compute_embeddings(model: Module, dataloaders: list[DataLoader]):
    for dataloader in dataloaders:
        pass

def get_backbone_from_ckpt(ckpt_path: str) -> torch.nn.Module:
    state_dict = torch.load(ckpt_path)["state_dict"]
    state_dict = OrderedDict([
        (".".join(name.split(".")[1:]), param) for name, param in state_dict.items() if name.startswith("backbone")
    ])

    return state_dict

class DomainMapper():
    def __init__(self, domains: List) -> None:
        self.unique_domains = domains.unique()
        self.map_dict: Dict = {self.unique_domains[i].item():i for i in range(len(self.unique_domains))}
        self.unmap_dict: Dict = dict((v, k) for k, v in self.map_dict.items())

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.map(x)

    def map(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([self.map_dict[v.item()] for v in x])
    
    def unmap(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([self.unmap_dict[v.item()] for v in x])