from typing import List, Dict
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, set_map: List[Dict], transform=None) -> None:
        ''' Each item in set_map is expected to contain:
                img_path: Full path to image,
                label: Label corresponding to image at img_path
        '''

        self.set_map = set_map
        self.transform=transform

    def __len__(self):
        return len(self.set_map)
    
    def __getitem__(self, index):   
        sample = self.set_map[index]
        image = Image.open(sample['img_path'])

        if self.transform:
            image = self.transform(image)

        return dict(image=image, **sample)