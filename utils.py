from typing import Iterator, List, Dict

import torch
from torch.utils.data import Sampler

class DeterministicSampler(Sampler):
    def __init__(self, data_source, initial_seed) -> None:
        self.data = data_source
        self.seed = initial_seed

    def __len__(self) -> int:
        return len(self.data)
    
    def len(self):
        return self.__len__()
    
    def __iter__(self) -> Iterator:
        print(self.seed)
        p = torch.randperm(self.len(), generator=torch.Generator().manual_seed(self.seed))
        # self.seed += 1
        r = torch.arange(self.len())[p]
        yield from r.tolist()

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