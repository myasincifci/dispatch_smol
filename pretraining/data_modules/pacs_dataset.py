import random
import pathlib
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.io import read_image

from PIL import Image

class PACSDataset(Dataset):
    def __init__(self, root: str, train=True, transform=None) -> None:
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
        self.paths = [str(p) for p in list(path.rglob("*.jpg"))]

        random.Random(42).shuffle(self.paths)

        if train:
            self.paths = self.paths[:int(len(self.paths)*.8)]
        else:
            self.paths = self.paths[int(len(self.paths)*.8):]


        self.transform = transform
        
    def __len__(self) -> int:
        return self.paths.__len__()

    def __getitem__(self, index) -> Tuple[torch.Tensor, int, int]:
        path = self.paths[index]
        image = read_image(path)
        domain = self.domains[path.split('/')[-3]]
        cls = self.classes[path.split('/')[-2]]

        if self.transform:
            image = self.transform(image)

        return (image, cls, domain)

def main():
    transform = T.ToTensor()

    dataset = PACSDataset('../data/PACS', True, transform)

    a=1

if __name__ == '__main__':
    from torchvision import transforms as T

    main()