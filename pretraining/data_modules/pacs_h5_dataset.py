import os
from typing import List, Any, Optional, Union, Dict
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.transforms.solarize import RandomSolarization

import random
import pathlib
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import numpy as np
from PIL.Image import Image
import h5py
import kornia.augmentation as K
from torchvision.transforms import v2 as T

from lightly.transforms.byol_transform import BYOLTransform

class BYOLViewTransform(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.4,
        cj_contrast: float = 0.4,
        cj_sat: float = 0.2,
        cj_hue: float = 0.1,
        min_scale: float = 0.08,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 1.0,
        solarization_prob: float = 0.0,
        kernel_size: Optional[float] = 1,
        sigmas: Tuple[float, float] = (0.1, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = 0.,
        normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):
        super().__init__()
        transform = [
            K.RandomResizedCrop(size=(input_size, input_size), scale=(min_scale, 1.0)),
            K.RandomRotation(p=rr_prob, degrees=rr_degrees),
            K.RandomHorizontalFlip(p=hf_prob),
            K.RandomVerticalFlip(p=vf_prob),
            K.ColorJitter(
                brightness=cj_strength*cj_bright, 
                contrast=cj_strength*cj_contrast, 
                saturation=cj_strength*cj_sat,  
                hue=cj_strength*cj_hue,
                p=cj_prob),
            K.RandomGrayscale(p=random_gray_scale),
            K.RandomGaussianBlur(kernel_size=(kernel_size,kernel_size), sigma=sigmas, p=gaussian_blur),
            K.RandomSolarize(p=solarization_prob, thresholds=0.5),
            T.ToTensor()
        ]
        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.transform = T.Compose(transform)

    def forward(self, image: Union[torch.Tensor, Image]) -> Tensor:
        return self.transform(image)

class PACSDataset(Dataset):
    def __init__(self, root: str, train=True, seed=42, transform=None) -> None:
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

        full_path = os.path.join(root, 'PACS.hdf5')
        self.f = h5py.File(full_path, 'r')
        self.D = self.f['D'][:]
        self.T = self.f['T'][:]

        N = len(self.f['X'])
        all_indices = torch.randperm(N)
        self.valid_indices = all_indices[:int(0.8*N)] if train else all_indices[int(0.8*N):]

        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitems__(self, indices):
        idcs = self.valid_indices[indices]

        _, index = torch.sort(idcs) # TODO: batch does not care if it is sorted
        sort = idcs[index] 
        sorted_images = torch.from_numpy(self.f['X'][sort])
        unsort = torch.arange(len(sorted_images)).gather(0, index.argsort(0))
        images = sorted_images[unsort]

        if self.transform:
            images = self.transform(images)

        domains = torch.from_numpy(self.f['D'][sort][unsort])
        clss = torch.from_numpy(self.f['T'][sort][unsort])

        return images, clss, domains

def main():
    transform = BYOLTransform(
        T.Compose([
            BYOLViewTransform(input_size=224),
        ]),
        T.Compose([
            BYOLViewTransform(
                    input_size=224,
                    gaussian_blur=0.1,
                    solarization_prob=0.2
            ),
        ])
    )

    dataset = PACSDataset(root='/data', transform=transform)
    (im1, im2), _, _ = dataset.__getitems__(torch.randperm(len(dataset))[:8])
    img1_grid = make_grid(im1)
    img2_grid = make_grid(im2)

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.imshow(img1_grid.permute(1,2,0))
    ax2.imshow(img2_grid.permute(1,2,0))

    plt.show()

    dl = DataLoader(dataset, batch_size=16, collate_fn=lambda x: x)
    a=1

if __name__ == '__main__':
    from torchvision.transforms import v2 as T
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    
    main()