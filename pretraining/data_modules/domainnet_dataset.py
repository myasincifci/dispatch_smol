import os
from typing import List, Dict
from PIL import Image
from torch.utils.data import Dataset
from .domainnet_metadata import DOMAIN_NET_CLASSES, DOMAIN_NET_DOMAINS

class ImageDataset(Dataset):
    def __init__(self, set_map: List[Dict], transform=None) -> None:
        ''' Each item in set_map is expected to contain:
                img_path: Full path to image,
                label: Label corresponding to image at img_path
        '''

        self.set_map = set_map
        self.transform = transform

    def __len__(self):
        return len(self.set_map)
    
    def __getitem__(self, index):   
        sample = self.set_map[index]

        image = Image.open(sample['img_path'])

        if self.transform:
            image = self.transform(image)

        return dict(image=image, **sample)

class DomainNetDataset(ImageDataset):
    def __init__(self, set_map: str, transform=None) -> None:
        super().__init__(set_map, transform)

        self.num_classes = len(DOMAIN_NET_CLASSES)
        self.num_domains = len(DOMAIN_NET_DOMAINS)
        self.class_map = dict(zip(DOMAIN_NET_CLASSES, range(len(DOMAIN_NET_CLASSES))))
        self.domain_map = dict(zip(DOMAIN_NET_DOMAINS, range(len(DOMAIN_NET_DOMAINS))))

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['label'] = self.class_map[item['label']]
        item['domain'] = self.domain_map[item['domain']]

        return item
