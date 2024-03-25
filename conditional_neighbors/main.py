from typing import Dict
import h5py
import torch
from tqdm import tqdm
from sklearn.neighbors import KDTree

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

    f = h5py.File('./neighborhood.hdf5', 'r')

    doms = {k.split('_')[-1] for k in f.keys()}
    # trees = {k: KDTree(z, leaf_size=1_000) for k, (z, _) in tqdm(doms.items())}
    trees: Dict[KDTree] = {}

    for d in tqdm(doms):
        trees[d] = KDTree(f[f'emb_{d}'], leaf_size=1_000)

    f2 = h5py.File('./neighborhood_lookup.hdf5', 'w')
    dset = f2.create_dataset('emb', (sum([len(f[f'emb_{d}']) for d in doms]), k*(len(doms)-1)), dtype='<i4')

    for d in doms:
        for i, d_ in enumerate(doms.symmetric_difference({d})):
            for z, z_idx_g in tqdm(zip(f[f'emb_{d}'], f[f'idx_{d}']), total=len(f[f'idx_{d}'])):
                # Convert local indices of neighbors to global
                local_idcs = trees[d_].query(z[None], k=k, return_distance=False).squeeze()
                global_idcs = f[f'idx_{d_}'][:][local_idcs].flatten()

                # Write to file at gloabl idx of d  
                dset[z_idx_g, i*k:(i+1)*k] = global_idcs[None]
                a=1

if __name__ == '__main__':
    main()