import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

TEST_CENTER = 2
VAL_CENTER = 1

class Camelyon17Dataset(Dataset):
    def __init__(self, root_dir='data', train=True, download=False):
        self._data_dir = os.path.join(root_dir, 'camelyon17_v1.0')
        self._original_resolution = (96,96)
        self._seed = 42

        # Read in metadata
        metadata = pd.read_csv(
            os.path.join(self._data_dir, 'metadata.csv'),
            index_col=0,
            dtype={'patient': 'str'})
        
        # 1. Get all entries with center 0 and 4
        mask = (metadata['center'] == 0) | (metadata['center'] == 4)
        metadata = metadata.loc[mask]

        # 2. Get 110k samples that are uniformly distibuted w.r.t the domain
        metadata = metadata.groupby('center', as_index=False).apply(lambda x: x.sample(n=55_000, random_state=self._seed))

        # 3. Do 80/20 train/test split
        train_df, test_df = train_test_split(metadata, test_size=0.2, random_state=42)
        metadata = train_df if train else test_df

        # 4. Apply 1:10 unbalancing for train samples
        if train:
            c0_t0 = metadata.loc[(metadata['center'] == 0) & (metadata['tumor'] == 0)].sample(n=10_000, random_state=self._seed)
            c0_t1 = metadata.loc[(metadata['center'] == 0) & (metadata['tumor'] == 1)].sample(n=1_000, random_state=self._seed)
            c4_t0 = metadata.loc[(metadata['center'] == 4) & (metadata['tumor'] == 0)].sample(n=1_000, random_state=self._seed)
            c4_t1 = metadata.loc[(metadata['center'] == 4) & (metadata['tumor'] == 1)].sample(n=10_000, random_state=self._seed)

            metadata = pd.concat([c0_t0, c0_t1, c4_t0, c4_t1])
        else: 
            metadata = metadata.sample(n=2_000, random_state=self._seed)

            c0_t0 = metadata.loc[(metadata['center'] == 0) & (metadata['tumor'] == 0)]
            c0_t1 = metadata.loc[(metadata['center'] == 0) & (metadata['tumor'] == 1)]
            c4_t0 = metadata.loc[(metadata['center'] == 4) & (metadata['tumor'] == 0)]
            c4_t1 = metadata.loc[(metadata['center'] == 4) & (metadata['tumor'] == 1)]

            a=1

        self._metadata_df = metadata

        # Get the y values
        self._y_array = torch.LongTensor(self._metadata_df['tumor'].values)
        self._d_array = torch.LongTensor(self._metadata_df['center'].values)
        self._y_size = 1
        self._n_classes = 2

        # Get filenames
        self._input_array = [
            f'patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png'
            for patient, node, x, y in
            self._metadata_df.loc[:, ['patient', 'node', 'x_coord', 'y_coord']].itertuples(index=False, name=None)]
        
        self._labels = torch.from_numpy(self._metadata_df['tumor'].to_numpy())
        self._domains = torch.from_numpy(self._metadata_df['center'].to_numpy())

        self._split_dict = {
            'train': 0,
            'test': 1,
        }
        self._split_names = {
            'train': 'Train',
            'test': 'Test',
        }

        

        # # Get domain masks
        # centers = self._metadata_df['center'].values.astype('long')

        # self._split_array = self._metadata_df['split'].values

        # self._metadata_array = torch.stack(
        #     (torch.LongTensor(centers),
        #      torch.LongTensor(self._metadata_df['slide'].values),
        #      self._y_array),
        #     dim=1)
        # self._metadata_fields = ['hospital', 'slide', 'y']

    def __len__(self):
        return len(self._input_array)

    def __getitem__(self, idx):
        img_filename = os.path.join(
            self.data_dir,
            self._input_array[idx])
        img = Image.open(img_filename).convert('RGB')
        label = 0
        domain = 0

        return dict(image=img, label=label, domain=domain)
    
def main():
    train_set = Camelyon17Dataset(root_dir='/data')
    test_set = Camelyon17Dataset(root_dir='/data', train=False)

    a=1

if __name__ == '__main__':
    main()