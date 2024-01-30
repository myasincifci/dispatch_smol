from typing import OrderedDict, List, Dict

import torch
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader

import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

from tqdm import tqdm

def compute_embeddings(model: nn.Module, source_loader, target_loader, grouper):
    BS = 32

    num_samples_source = source_loader.dataset.__len__()
    num_samples_target = target_loader.dataset.__len__()

    source = {
        'z0': torch.empty((num_samples_source, 64)),
        'z1': torch.empty((num_samples_source, 64)),
        'z2': torch.empty((num_samples_source, 128)),
        'z3': torch.empty((num_samples_source, 256)),
        'z4': torch.empty((num_samples_source, 512)),
        'x': torch.empty((num_samples_source, 512)),
        't': torch.empty((num_samples_source)),
        'd': torch.empty((num_samples_source)),
    }
    target = {
        'z0': torch.empty((num_samples_target, 64)),
        'z1': torch.empty((num_samples_target, 64)),
        'z2': torch.empty((num_samples_target, 128)),
        'z3': torch.empty((num_samples_target, 256)),
        'z4': torch.empty((num_samples_target, 512)),
        'x': torch.empty((num_samples_target, 512)),
        't': torch.empty((num_samples_target)),
        'd': torch.empty((num_samples_target)),
    }

    with torch.no_grad():
        for i, (X, t, M) in enumerate(tqdm(source_loader)):
            bs = len(X)

            d = grouper.metadata_to_group(M)

            embs = model.embed(X.cuda())
            source['z0'][i*BS:i*BS+bs] = embs['z0'].cpu()
            source['z1'][i*BS:i*BS+bs] = embs['z1'].cpu()
            source['z2'][i*BS:i*BS+bs] = embs['z2'].cpu()
            source['z3'][i*BS:i*BS+bs] = embs['z3'].cpu()
            source['z4'][i*BS:i*BS+bs] = embs['z4'].cpu()
            source['x'][i*BS:i*BS+bs] = embs['x'].cpu()
            source['t'][i*BS:i*BS+bs] = t
            source['d'][i*BS:i*BS+bs] = d

        for i, (X, t, M) in enumerate(tqdm(target_loader)):
            bs = len(X)

            d = grouper.metadata_to_group(M)

            embs = model.embed(X.cuda())
            target['z0'][i*BS:i*BS+bs] = embs['z0'].cpu()
            target['z1'][i*BS:i*BS+bs] = embs['z1'].cpu()
            target['z2'][i*BS:i*BS+bs] = embs['z2'].cpu()
            target['z3'][i*BS:i*BS+bs] = embs['z3'].cpu()
            target['z4'][i*BS:i*BS+bs] = embs['z4'].cpu()
            target['x'][i*BS:i*BS+bs] = embs['x'].cpu()
            target['t'][i*BS:i*BS+bs] = t
            target['d'][i*BS:i*BS+bs] = d

    return source, target

def get_backbone_from_ckpt(ckpt_path: str) -> torch.nn.Module:
    state_dict = torch.load(ckpt_path)["state_dict"]
    state_dict = OrderedDict([
        (".".join(name.split(".")[2:]), param) for name, param in state_dict.items() if name.startswith("model.backbone")
    ])

    return state_dict

class DomainMapper():
    def __init__(self, domains: List) -> None:
        self.unique_domains = domains.unique()
        self.map_dict: Dict = {self.unique_domains[i].item():i for i in range(len(self.unique_domains))}
        self.unmap_dict: Dict = dict((v, k) for k, v in self.map_dict.items())

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.map(x)

    def map(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([self.map_dict[v.item()] for v in x])
    
    def unmap(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([self.unmap_dict[v.item()] for v in x])
    
################################################################################
def plot_layer(ax, source: torch.Tensor, target: torch.Tensor, layer='x', acc: float=0.0):
    reducer = umap.UMAP(n_neighbors=5, n_epochs=1000, min_dist=0.1)
    all_embeddings = torch.cat([source[layer], target[layer]],dim=0)
    all_targets = torch.cat([source['t'], target['t']], dim=0)
    
    X_scaled = StandardScaler().fit_transform(all_embeddings)

    reducer = umap.UMAP(n_neighbors=5, min_dist=0.07)
    X_reduced = reducer.fit_transform(X_scaled)

    source_target = torch.cat([torch.zeros(10_000), torch.ones(10_000)])

    color_values = torch.cat([source_target[None,:], all_targets[None,:]], dim=0)

    color_map = {
        (0,0): "royalblue", # source, no_tumor
        (0,1): "mediumaquamarine", # source, tumor
        (1,0): "gold", # target, no_tumor
        (1,1): "goldenrod" # target, tumor
    }

    colors_master = [color_map[tuple(x.to(int).tolist())] for x in color_values.T]

    ax.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        c=colors_master,
        s=0.05
    )
    ax.set_xlabel(f"Layer: {layer}\n Accuracy: {acc}")

def predict_domain(source, target, layer='x'):
    X = torch.cat([source[layer], target[layer]], dim=0)
    t = torch.cat([source['d'], target['d']], dim=0)

    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ('scaler', StandardScaler()), 
        ('lor', LogisticRegression(max_iter=400, C=0.01))
    ])

    pipe.fit(X_train, t_train)
    # print(classification_report(t_test, pipe.predict(X_test)))

    return accuracy_score(t_test, pipe.predict(X_test))

def evaluate(model: nn.Module, source_loader, target_loader, grouper):
    source, target = compute_embeddings(model, source_loader, target_loader, grouper)

    # Compute domain accuracy
    acc_l0 = 0# predict_domain(source, target, 'z0')
    acc_l1 = 0# predict_domain(source, target, 'z1')
    acc_l2 = 0# predict_domain(source, target, 'z2')
    acc_l3 = 0# predict_domain(source, target, 'z3')
    acc_l4 = 0# predict_domain(source, target, 'z4')

    # Plot embeddings
    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1,5)

    fig.set_figheight(3)
    fig.set_figwidth(18)

    plot_layer(ax0, source, target, layer='z0', acc=acc_l0)
    plot_layer(ax1, source, target, layer='z1', acc=acc_l1)
    plot_layer(ax2, source, target, layer='z2', acc=acc_l2)
    plot_layer(ax3, source, target, layer='z3', acc=acc_l3)
    plot_layer(ax4, source, target, layer='z4', acc=acc_l4)