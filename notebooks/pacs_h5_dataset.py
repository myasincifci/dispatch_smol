from typing import List

import random
import pathlib
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image
import h5py

class PACSDataset(Dataset):
    def __init__(self, root: str, leave_out: List=None, transform=None) -> None:
        self.classes = {
            'dog': 0, 
            'giraffe': 1, 
            'guitar': 2, 
            'house': 3, 
            'person': 4, 
            'horse': 5, 
            'elephant': 6
        }

        self.domains = {
            'sketch': 0, 
            'cartoon': 1, 
            'art_painting': 2, 
            'photo': 3
        }

        self.n_classes = len(self.classes)

        path = pathlib.Path(root)
        self.f = h5py.File(path, 'r')
        self.D = self.f['D'][:]
        self.T = self.f['T'][:]

        indices = torch.arange(len(self.f['X']))
        holdout = [self.domains[l] for l in leave_out]

        index_mask = torch.ones_like(indices)
        for i in holdout:
            index_mask = index_mask & (self.D != i)
        index_mask = index_mask.to(torch.bool)

        self.valid_indices = indices[index_mask]

        self.transform = transform
        
    def __len__(self) -> int:
        return self.valid_indices.__len__()
    
    def __getitems__(self, idcs):
        idcs_ = self.valid_indices[idcs].numpy()

        images = self.f['X'][idcs_]
        domains = self.f['D'][idcs_]
        clss = self.f['T'][idcs_]

        return list(zip(images, clss, domains))
    
def get_pacs_loo(leave_out=None):
    pass

def main():
    dataset = PACSDataset('./notebooks/PACS.hdf5', leave_out=['sketch', 'cartoon'])

    dataset.__getitems__([0,1,3,5,8,9])

    a=1

if __name__ == '__main__':
    main()