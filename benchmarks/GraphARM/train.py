from torch_geometric.datasets import ZINC
from tqdm import tqdm
import torch
from torch import nn
import math
import wandb
import os

from benchmarks.GraphARM.models import DiffusionOrderingNetwork, DenoisingNetwork
from benchmarks.GraphARM.utils import NodeMasking
from benchmarks.GraphARM.grapharm import GraphARM

# instanciate the dataset
dataset = ZINC(root='~/workspace/GraphDiffusionImitate/data/ZINC', transform=None, pre_transform=None)

diff_ord_net = DiffusionOrderingNetwork(node_feature_dim=1,
                                        num_node_types=dataset.x.unique().shape[0],
                                        num_edge_types=2,
                                        num_layers=3,
                                        out_channels=1)

masker = NodeMasking(dataset)


denoising_net = DenoisingNetwork(
    node_feature_dim=dataset.num_features,
    edge_feature_dim=dataset.num_edge_features,
    num_node_types=dataset.x.unique().shape[0],
    num_edge_types=2,
    num_layers=7,
    out_channels=1
)

wandb.init(
        project="ARGD",
        group=f"v1.3.0",
        name=f"ZINC",
        # track hyperparameters and run metadata
        config={
            "policy": "train",
            "n_epochs": 10000,
            "batch_size": 1,
        }
    )

os.environ["WANDB_DISABLED"] = "false"
torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

grapharm = GraphARM(
    dataset=dataset,
    denoising_network=denoising_net,
    diffusion_ordering_network=diff_ord_net,
    device=device
)

batch_size = 5
try:
    grapharm.load_model()
    print("Loaded model")
except:
    print ("No model to load")
# train loop
for epoch in range(2000):
    print(f"Epoch {epoch}")
    grapharm.train_step(
        train_data=dataset[2*epoch*batch_size:(2*epoch + 1)*batch_size],
        val_data=dataset[(2*epoch + 1)*batch_size:batch_size*(2*epoch + 2)],
        M=4
    )
    grapharm.save_model()