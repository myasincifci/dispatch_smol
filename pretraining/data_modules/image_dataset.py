import os
import copy
from typing import List, Dict, Optional
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, data_path, set_map: List[Dict], transform=None, 
                 classes: Optional[Dict[str, int]] = None,
                 domains: Optional[Dict[str, int]] = None) -> None:
        ''' Each item in set_map is expected to contain:
                img_path: Full path to image,
                label: Label corresponding to image at img_path
                domain: Domain corresponding to image at img_path
        '''

        self.set_map = copy.deepcopy(set_map)
        self.transform=transform

        if classes:
            for sample in self.set_map:
                sample['label'] = classes[sample['label']]
        if domains:
            for sample in self.set_map:
                sample['domain'] = domains[sample['domain']]

        self.classes = classes
        self.doamins = domains
        self.data_path = data_path

    def __len__(self):
        return len(self.set_map)
    
    def __getitem__(self, index):   
        sample = self.set_map[index]
        image = Image.open(os.path.join(self.data_path, sample['img_path']))

        if self.transform:
            image = self.transform(image)

        return dict(image=image, **sample)