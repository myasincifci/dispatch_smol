import numpy as np
from wilds.datasets.wilds_dataset import WILDSSubset
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset

class WILDSSubsetMod(WILDSSubset):
    def __getitem__(self, idx):
        x, t, m =  super().__getitem__(idx)
        d = m[0]

        # Get x_ by uniformly sampling from indices that have the same class label but not domain
        ds: Camelyon17DatasetMod = self.dataset
        
        I = self.indices # Indices of subset
        D = ds.metadata_array[I,0]
        T = ds.y_array[I]
        R = np.arange(len(I))

        idx_ = np.random.choice(R[(D != d)&(T == t)], 1)

        x_, _, _ = super().__getitem__(idx_.item())

        return {
            'x': x,
            'x_': x_,
            't': t,
            'm': m
        }
    
class Camelyon17DatasetMod(Camelyon17Dataset):
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

        return WILDSSubsetMod(self, split_idx, transform)

def main():
    dataset = Camelyon17DatasetMod(root_dir='../data/')
    train_set = dataset.get_subset('train')

    train_set.__getitem__(0)

    a=1

if __name__ == '__main__':
    main()