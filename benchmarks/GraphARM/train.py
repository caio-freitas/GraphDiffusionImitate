from torch_geometric.datasets import ZINC
from tqdm import tqdm
import torch
from torch import nn
import math
import wandb

from benchmarks.GraphARM.models import DiffusionOrderingNetwork, DenoisingNetwork
from benchmarks.GraphARM.utils import NodeMasking
from benchmarks.GraphARM.grapharm import GraphARM

# instanciate the dataset
dataset = ZINC(root='~/workspace/GraphDiffusionImitate/data/ZINC', transform=None, pre_transform=None)

diff_ord_net = DiffusionOrderingNetwork(node_feature_dim=1,
                                        num_node_types=dataset.x.unique().shape[0] + 1,
                                        num_edge_types=3,
                                        num_layers=3,
                                        out_channels=1)

masker = NodeMasking(dataset)


denoising_net = DenoisingNetwork(
    node_feature_dim=dataset.num_features,
    num_node_types=dataset.x.unique().shape[0] + 1,
    num_edge_types=3,
    num_layers=7,
    out_channels=1
)

wandb.init(
        project="ARGD",
        group=f"v1.0.1",
        name=f"denoising_and_ordering",
        # track hyperparameters and run metadata
        config={
            "policy": "train",
            "n_epochs": 10000,
            "batch_size": 1,
        }
    )


grapharm = GraphARM(
    dataset=dataset,
    denoising_network=denoising_net,
    diffusion_ordering_network=diff_ord_net
)

batch_size = 5
# train loop
for epoch in range(100):
    print(f"Epoch {epoch}")
    grapharm.train_step(
        train_data=dataset[epoch*batch_size:(epoch + 1)*batch_size],
        val_data=dataset[(epoch + 1)*batch_size:batch_size*(epoch + 2)],
        M=4
    )
    grapharm.save_model()