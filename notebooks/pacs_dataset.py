import random
import pathlib
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from PIL import Image

class PACSDataset(Dataset):
    def __init__(self, root: str, split='train', transform=None) -> None:
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
        self.paths = [str(p) for p in list(path.rglob("*.[jp][pn]g"))]

        random.Random(42).shuffle(self.paths)

        if split == 'train':
            self.paths = self.paths[:int(len(self.paths)*.8)]
        elif split == 'test':
            self.paths = self.paths[int(len(self.paths)*.8):]
        elif split == 'full':
            pass
        else:
            raise Exception('Invalid Split')


        self.transform = transform
        
    def __len__(self) -> int:
        return self.paths.__len__()

    def __getitem__(self, index) -> Tuple[torch.Tensor, int, int]:
        path = self.paths[index]
        image = Image.open(path)
        domain = self.domains[path.split('/')[-3]]
        cls = self.classes[path.split('/')[-2]]

        if self.transform:
            image = self.transform(image)

        return (image, cls, domain)

def make_hdf5():
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import h5py

    dataset = PACSDataset('../data/PACS', split='full', transform=T.ToTensor())
    dataloader = DataLoader(dataset, 1)

    f = h5py.File('PACS.hdf5', 'w')
    f.create_dataset('X', (len(dataset), 3, 227, 227))
    f.create_dataset('T', (len(dataset),))
    f.create_dataset('D', (len(dataset),))

    for i, batch in enumerate(tqdm(dataloader)):
        X, t, d = batch
        
        f['X'][i] = X
        f['T'][i] = t
        f['D'][i] = d

    f.close()

def main():
    transform = T.ToTensor()

    dataset = PACSDataset('../data/PACS', split='full', transform=transform)

    a=1

if __name__ == '__main__':
    from torchvision import transforms as T

    main()