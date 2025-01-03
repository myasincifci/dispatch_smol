import os

import torch
from torch.utils.data import Dataset
from PIL import Image

class DRDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, selected_domains=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.domains = {
            'aptos': 0,
            'eyepacs': 1,
            'messidor': 2,
            'messidor_2': 3
        }
        self.classes = [0, 1, 2, 3, 4]
        self.selected_domains = list(self.domains.keys()) if not selected_domains else selected_domains

        for i, domain in enumerate(self.selected_domains):
            self.domains[domain] = i
            domain_path = os.path.join(root_dir, domain)
            
            for class_label in self.classes:
                class_path = os.path.join(domain_path, str(class_label))
                
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    self.samples.append((img_path, domain, class_label))

        self.n_classes = len(self.classes)

    def __len__(self):
        return len(self.samples) 

    def __getitem__(self, index):
        img_path, domain, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, torch.tensor([self.domains[domain]])
    
def get_loo_dr(root, leave_out: str, train_tf, test_tf):
    ''' Build Leave One Out Datasets
    '''
    all_domain = set(['aptos', 'eyepacs', 'messidor', 'messidor_2'])
    train_domains = list(all_domain - set([leave_out]))
    test_domains = [leave_out]

    train_set = DRDataset(root_dir=root, transform=train_tf, selected_domains=train_domains)
    test_set = DRDataset(root_dir=root, transform=test_tf, selected_domains=test_domains)

    return train_set, test_set

if __name__ == '__main__':
    from torchvision.transforms import v2 as T

    # Example usage
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    train_set, test_set = get_loo_dr('/data/DR', leave_out='aptos', train_tf=transform, test_tf=transform)

    a=1