
from tqdm import tqdm
import torch
import wandb
import torch.nn as nn

from benchmarks.GraphARM.models import DiffusionOrderingNetwork, DenoisingNetwork
from benchmarks.GraphARM.utils import NodeMasking


EPSILON = 1e-8


class GraphARM(nn.Module):
    '''
    Class to encapsule DiffusionOrderingNetwork and DenoisingNetwork, as well as the training loop
    for both with diffusion and denoising steps.
    '''
    def __init__(self,
                 dataset,
                 denoising_network,
                 diffusion_ordering_network,):
        super(GraphARM, self).__init__()

        self.diffusion_ordering_network = diffusion_ordering_network

        self.denoising_network = denoising_network
        self.masker = NodeMasking(dataset)


        self.denoising_optimizer = torch.optim.Adam(self.denoising_network.parameters(), lr=1e-4, betas=(0.9, 0.999))
        self.ordering_optimizer = torch.optim.Adam(self.diffusion_ordering_network.parameters(), lr=5e-4, betas=(0.9, 0.999))


    def node_decay_ordering(self, datapoint):
        p = datapoint.clone()
        node_order = []
        for i in range(p.x.shape[0]):
            # use diffusion ordering network to get probabilities
            sigma_t_dist = self.diffusion_ordering_network(p, i)
            # sample from categorical distribution to get node to mask
            # TODO only on the samples that are not masked
            unmasked = ~self.masker.is_masked(p)
            sigma_t = torch.distributions.Categorical(probs=sigma_t_dist[unmasked].flatten()).sample()
            
            # get node index
            sigma_t = torch.where(unmasked.flatten())[0][sigma_t.long()]
            node_order.append(sigma_t)
            # mask node
            p = self.masker.mask_node(p, sigma_t)
        return node_order

    def generate_diffusion_trajectories(self, graph, M):
        '''
        Generates M diffusion trajectories for a given graph,
        using the node decay ordering mechanism.
        '''
        original_data = graph.clone()
        diffusion_trajectories = []
        for m in range(M):
            node_order = self.node_decay_ordering(graph)
            
            # create diffusion trajectory
            diffusion_trajectory = [original_data]
            masked_data = graph.clone()
            for node in node_order:
                masked_data = masked_data.clone()
                
                masked_data = self.masker.mask_node(masked_data, node)
                diffusion_trajectory.append(masked_data)

            diffusion_trajectories.append(diffusion_trajectory)
        return diffusion_trajectories

    def train_step(
            self,
            train_data,
            val_data,
            M = 4, # number of diffusion trajectories to be created for each graph
        ):

        self.denoising_network.train()
        self.diffusion_ordering_network.eval()
        loss = torch.tensor(0.0, requires_grad=True)
        with tqdm(train_data) as pbar:
            for graph in pbar:
                diffusion_trajectories = self.generate_diffusion_trajectories(graph, M)  
                # predictions & loss
                for diffusion_trajectory in diffusion_trajectories:
                    G_0 = diffusion_trajectory[0]
                    # optimizer.zero_grad()
                    node_order = self.node_decay_ordering(G_0) # TODO check. This isn't available when sampling
                    for t in range(1, len(diffusion_trajectory)-1):
                        node = node_order[len(diffusion_trajectory)-t-1]
                        G_t = diffusion_trajectory[t].clone()
                        G_tplus1 = diffusion_trajectory[t+1].clone()
                        G_pred = G_tplus1.clone()
                        

                        # predict node type
                        node_type_probs, edge_type_probs = self.denoising_network(G_pred)
                        
                        # calculate loss
                        p_O_v =  node_type_probs[node].mean() + EPSILON # TODO add edges (joint probability)
                        w_k = self.diffusion_ordering_network(G_tplus1)[node]
                        n_i = G_t.x.shape[0]
                        # TODO check this. G_t isn't even being used. 
                        loss = loss - (n_i/(len(diffusion_trajectory)-1))*torch.log(p_O_v)*w_k/M
        
        # backprop
        loss.backward()
        # update parameters
        self.denoising_optimizer.step()
        
        # log loss
        pbar.set_description(f"Loss: {loss.item()%10:.4f}")
        wandb.log({"loss": loss.item()})


        # validation batch (for diffusion ordering network)
        self.denoising_network.eval()
        self.diffusion_ordering_network.train()

        reward = torch.tensor(0.0, requires_grad=True)
        with tqdm(val_data) as pbar:
            for graph in pbar:
                n_i = graph.x.shape[0]
                
                diffusion_trajectories = self.generate_diffusion_trajectories(graph, M)

                for diffusion_trajectory in diffusion_trajectories:
                    G_0 = diffusion_trajectory[0]
                    node_order = self.node_decay_ordering(G_0)
                    for t in range(len(diffusion_trajectory)-1):
                        node = node_order[G_0.x.shape[0] - t - 1]
                        G_tplus1 = diffusion_trajectory[t+1]
                        # predict node type
                        node_type_probs, edge_type_probs = self.denoising_network(G_tplus1)

                        # calculate reward (VLB)                        
                        p_O_v =  node_type_probs[node].mean() + EPSILON # TODO add edges (joint probability)

                        # reward -= torch.log(O_v)*n_i/(len(diffusion_trajectory)-1)
                        r = (n_i/(len(diffusion_trajectory)-1))*torch.log(p_O_v)
                        w_k = self.diffusion_ordering_network(G_tplus1)[node]

                        reward = reward - w_k*r/M

        
        wandb.log({"reward": reward.item()})
        # update parameters (REINFORCE algorithm)
        self.ordering_optimizer.zero_grad()
        reward.backward()
        self.ordering_optimizer.step()
        pbar.set_description(f"Reward: {reward.item()%10:.4f}")


    def save_model(self):
        torch.save(self.denoising_network.state_dict(), "denoising_network.pt")
        torch.save(self.diffusion_ordering_network.state_dict(), "diffusion_ordering_network.pt")