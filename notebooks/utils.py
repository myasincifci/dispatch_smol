from torch.nn import Module
from torch.utils.data import DataLoader

def compute_embeddings(model: Module, dataloaders: list[DataLoader]):
    for dataloader in dataloaders:
        pass