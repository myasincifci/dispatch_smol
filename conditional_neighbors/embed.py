import torch
from tqdm import tqdm
from sklearn.neighbors import KDTree
import h5py
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from torchvision import transforms as T

from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.unlabeled.camelyon17_unlabeled_dataset import Camelyon17UnlabeledDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.wilds_dataset import WILDSSubset
from wilds.datasets.unlabeled.wilds_unlabeled_dataset import WILDSUnlabeledSubset

from models.model import BarlowTwins

from torchvision.models.resnet import resnet50

import numpy as np

BS = 128

class Camelyon17DatasetIdx(Camelyon17UnlabeledDataset):
    def get_subset(self, split, frac=1, transform=None):
        """
        Args:
            - split (str): Split identifier, e.g., 'train', 'val', 'test'.
                           Must be in self.split_dict.
            - frac (float): What fraction of the split to randomly sample.
                            Used for fast development on a small dataset.
            - transform (function): Any data transformations to be applied to the input x.
        Output:
            - subset (WILDSSubset): A (potentially subsampled) subset of the WILDSDataset.
        """
        if split not in self.split_dict:
            raise ValueError(f"Split {split} not found in dataset's split_dict.")

        split_mask = self.split_array == self.split_dict[split]
        split_idx = np.where(split_mask)[0]

        if frac < 1.0:
            # Randomly sample a fraction of the split
            num_to_retain = int(np.round(float(len(split_idx)) * frac))
            split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])

        return WILDSSubsetIdx(self, split_idx, transform)
    
class WILDSSubsetIdx(WILDSUnlabeledSubset):
    def __getitem__(self, idx):
        x, m =  super().__getitem__(idx)

        return {
            'idx': idx,
            'x': x,
            'm': m
        }

def main():
    dataset = Camelyon17DatasetIdx(root_dir='/data')
    grouper = CombinatorialGrouper(dataset, ['hospital'])

    train_set = dataset.get_subset('train_unlabeled', transform=T.Compose([T.Resize(96), T.ToTensor()]))
    train_loader = get_train_loader('standard', train_set, batch_size=BS)

    # model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
    # model.fc = torch.nn.Identity()
    # model = model.cuda()
    # TODO: ugly bc of bug in lightning, fix later 
    weights = torch.load('/home/myasincifci/dispatch_smol/conditional_neighbors/models/epoch=28-step=50000.ckpt', map_location=torch.device('cpu'))['state_dict']
    bt = BarlowTwins(None, resnet50(), None, None, None)
    bt.load_state_dict(weights, strict=False)
    model = bt.backbone
    model.fc = torch.nn.Identity()
    model = model.cuda()

    # Create one dataset per domain and save with global indices
    f = h5py.File('neighborhood.hdf5', 'w')
    counter = {}
    for i, n in enumerate(torch.bincount(train_set.metadata_array[:,0])):
        if n == 0:
            pass
        else:
            counter[i] = 0
            f.create_dataset(f'emb_{i}', (n, 2048), dtype='f')
            f.create_dataset(f'idx_{i}', (n, 1), dtype='i')

    with torch.no_grad():
        for batch in tqdm(train_loader):
            idx, x, m = batch['idx'], batch['x'], batch['m']
            d = grouper.metadata_to_group(m)

            z = model(x.cuda())
            
            for d_ in torch.unique(d):
                occ = d == d_
                count = occ.sum().item()

                z_d = z[occ]
                idx_d = idx[occ]

                start = counter[d_.item()]
                stop = start + count

                f[f'emb_{d_}'][start:stop] = z_d[:].cpu().detach() # torch.arange(counter[d_.item()], counter[d_.item()]+count)[:,None].repeat(1, 2048)
                f[f'idx_{d_}'][start:stop] = idx_d[:,None]

                counter[d_.item()] += count

        a=1

if __name__ == '__main__':
    main()