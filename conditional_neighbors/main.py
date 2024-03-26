from typing import Dict
import h5py
import torch
import numpy as np
from tqdm import tqdm
from itertools import batched
import math

# from sklearn.neighbors import KDTree
from scipy.spatial import KDTree

def nearest_ood_neighbors(z: torch.Tensor, k: int, d: int):
    """Returns global indices of k n-neighbors per domain in all domains 
    that are not d (reurns (D-1)*n neighbors in total).

    Parameters
    ----------
    z : torch.Tensor
        Embedding for which to find the nearest neighbors.
    d : int
        ID of domain from which z originates.
    k : int
        Number of neighbors to be returned 
    """   

    idcs = [] 
    (g := list(doms.keys())).remove(d)
    for d_ in g:
        local_idcs = trees[d_].query(z, k=k, return_distance=False)
        global_idcs = [doms[d_][1][i] for i in local_idcs]
        idcs.append(global_idcs[0])

    return torch.hstack(idcs)

def main():
    k = 10
    bs = 128

    f = h5py.File('/tmp/neighborhood.hdf5', 'r')

    doms = {k.split('_')[-1] for k in f.keys()}
    trees: Dict[KDTree] = {}

    for d in tqdm(doms):
        trees[d] = KDTree(f[f'emb_{d}'], leafsize=1_000)

    f2 = h5py.File('/tmp/neighborhood_lookup.hdf5', 'w')
    dset = f2.create_dataset('emb', (sum([len(f[f'emb_{d}']) for d in doms]), k*(len(doms)-1)), dtype='<i4')

    for d in sorted(doms):
        for i, d_ in enumerate(sorted(doms.symmetric_difference({d}))):
            for batch in tqdm(batched(zip(f[f'emb_{d}'], f[f'idx_{d}']), n=bs), total=math.ceil(len(f[f'idx_{d}'])/bs)):
                z, z_idx_g = np.vstack([e for e, _ in batch]), np.vstack([i for _, i in batch]).squeeze()

                # Convert local indices of neighbors to global
                _, local_idcs = trees[d_].query(z, k=k, workers=4)
                local_idcs = local_idcs.squeeze()
                global_idcs = f[f'idx_{d_}'][:][local_idcs].squeeze()

                # Write to file at gloabl idx of d  
                sorting = z_idx_g.argsort()
                dset[z_idx_g[sorting], i*k:(i+1)*k] = global_idcs[sorting]

if __name__ == '__main__':
    main()