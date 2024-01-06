'''
Diffusion Policy for imitation learning with graphs
'''
import torch
from tqdm import tqdm
import torch.nn as nn
import wandb

from imitation.model.graph_diffusion import DiffusionOrderingNetwork, DenoisingNetwork
from imitation.utils.graph_diffusion import NodeMasker
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler


class GraphDiffusionPolicy(nn.Module):
    def __init__(self,
                 dataset,
                 node_feature_dim,
                 num_edge_types,
                 denoising_network,
                 lr=1e-4,
                 ckpt_path=None,
                 device='cpu'):
        super(GraphDiffusionPolicy, self).__init__()
        self.dataset = dataset
        self.device = device
        self.node_feature_dim = node_feature_dim
        self.num_edge_types = num_edge_types
        self.denoising_network = denoising_network.to(self.device)
        # no need for diffusion ordering network
        self.node_feature_loss = nn.MSELoss()
        self.masker = NodeMasker(dataset)
        

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5)

        # self.dataloader = torch.utils.data.DataLoader(
        #     self.dataset,
        #     batch_size=1,
        #     num_workers=1,
        #     shuffle=True,
        #     # accelerate cpu-gpu transfer
        #     pin_memory=True,
        #     # don't kill worker process afte each epoch
        #     persistent_workers=True
        # )

    def generate_diffusion_trajectory(self, graph):
        '''
        Generate a diffusion trajectory from graph
        - Diffuse nodes representing object first
        - Then diffuse nodes representing robot joints, from end-effector to base
        '''
        diffusion_trajectory = []
        '''
            Initially permutation variant, since node ordering is known (robot nodes first, then object node(s))
        '''
        node_order = torch.arange(graph.x.shape[0]-1, -1, -1)
        diffusion_trajectory.append(graph)
        masked_data = graph.clone()
        for t in range(graph.x.shape[0]):
            node = node_order[t]
            masked_data = masked_data.clone().to(self.device)
            
            masked_data = self.masker.mask_node(masked_data, node)
            diffusion_trajectory.append(masked_data)
            # don't remove last node
            if t < len(node_order)-1:
                masked_data = self.masker.remove_node(masked_data, node)
                node_order = [n-1 if n > node else n for n in node_order] # update node order to account for removed node

        return diffusion_trajectory

    def preprocess(self, graph):
        '''
        Preprocesses graph to be used by the denoising network.
        '''
        graph = graph.clone()
        # edge types to idx
        graph = self.masker.idxify(graph)
        graph = self.masker.fully_connect(graph)
        return graph

    def train(self, dataset, num_epochs=100, model_path=None, seed=0):
        '''
        Train noise prediction model
        '''
        self.optimizer.zero_grad()
        self.denoising_network.train()


        # batch loop
        with tqdm(dataset, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                print(nbatch)
                # preprocess graph
                graph = self.preprocess(nbatch)
                diffusion_trajectory = self.generate_diffusion_trajectory(graph)  
                # predictions & loss
                G_0 = diffusion_trajectory[0]
                node_order = self.node_decay_ordering(G_0)
                for t in range(len(node_order)):
                    for k in range(t+1):# until t
                        G_pred = diffusion_trajectory[t+1].clone()                              

                        # predict node and edge type distributions
                        node_features, edge_type_probs = self.denoising_network(G_pred.x, G_pred.edge_index, G_pred.edge_attr)

                        w_k = self.diffusion_ordering_network(G_0, node_order[t+1:])[node_order[k]]
                        w_k = w_k.detach()
                        wandb.log({"target_node_ordering_prob": w_k.item()})
                        # calculate loss relative to edge type distribution
                        loss = self.vlb(G_0, edge_type_probs, w_k, node_order[k], node_order, t) # cumulative loss
                        # mse loss for node features
                        node_loss = self.node_feature_loss(node_features, G_pred.x[node_order[k]])
                        wandb.log({"vlb": loss.item(), "node_feature_loss": node_loss.item()})
                        loss += node_loss
                        wandb.log({"loss": loss.item()})

                        acc_loss += loss.item()
                        # backprop (accumulated gradients)
                        loss.backward()
                        pbar.set_description(f"Loss: {acc_loss:.4f}")
