from tqdm import tqdm
import torch
import torch_geometric
from torch_geometric.datasets import ZINC
from torch import nn
import math
import wandb
import os
import logging
logging.basicConfig(level=logging.DEBUG)

from benchmarks.GraphARM.models import DiffusionOrderingNetwork, DenoisingNetwork
from benchmarks.GraphARM.utils import NodeMasking
from benchmarks.GraphARM.grapharm import GraphARM


# Create example data.Data with torch_geometric to overfit to
# data = torch_geometric.data.Data(
#     x=torch.tensor([[4],
#                     [4],
#                     [1],
#                     [1],
#                     [1]]),
#     edge_index=torch.tensor([[0, 1, 2, 3, 4], 
#                              [1, 2, 3, 4, 0]]),
#     edge_attr=torch.tensor([[1.0], 
#                             [1.0], 
#                             [2.0], 
#                             [3.0], 
#                             [3.0]]),
# )
# dataset = data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

dataset = ZINC(root='~/workspace/GraphDiffusionImitate/data/ZINC', transform=None, pre_transform=None)
data = dataset[1]

diff_ord_net = DiffusionOrderingNetwork(node_feature_dim=1,
                                        num_node_types=dataset.x.unique().shape[0],
                                        num_edge_types=dataset.edge_attr.unique().shape[0],
                                        num_layers=3,
                                        hidden_dim=16,
                                        out_channels=1,
                                        device=device)


masker = NodeMasking(dataset)


denoising_net = DenoisingNetwork(
    node_feature_dim=dataset.num_features,
    edge_feature_dim=dataset.num_edge_features,
    num_node_types=5,
    num_edge_types=dataset.edge_attr.unique().shape[0],
    num_layers=7,
    hidden_dim=32,
    device=device
    )

wandb.init(
        project="GraphARM",
        group=f"v2.3.0",
        name=f"ZINC_DON_overfit",
        config={
            "policy": "train",
            "n_epochs": 10000,
            "batch_size": 1,
            "lr": 1e-3,
        },
        mode='disabled'
    )


torch.autograd.set_detect_anomaly(True)

grapharm = GraphARM(
    dataset=dataset,
    denoising_network=denoising_net,
    diffusion_ordering_network=diff_ord_net,
    device=device
)

dataset = [data]

batch_size = 5
try:
    grapharm.load_model("~/workspace/GraphDiffusionImitate/MOCK_overfit_pooling_message_passing.pt")
    print("Loaded model")
except:
    print ("No model to load")
    # raise
# train loop
for epoch in range(1000):
    print(f"Epoch {epoch}")
    grapharm.train_step(
        train_data=dataset,
        val_data=dataset,
        M=4
    )
    grapharm.save_model("MOCK_overfit_pooling_message_passing.pt")