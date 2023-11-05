
from tqdm import tqdm
import torch
import wandb
import torch.nn as nn

from benchmarks.GraphARM.models import DiffusionOrderingNetwork, DenoisingNetwork
from benchmarks.GraphARM.utils import NodeMasking

EPSILON = 1e-10


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
            # sample (only unmasked nodes) from categorical distribution to get node to mask
            unmasked = ~self.masker.is_masked(p)
            sigma_t = torch.distributions.Categorical(probs=sigma_t_dist[unmasked].flatten()).sample()
            
            # get node index
            sigma_t = torch.where(unmasked.flatten())[0][sigma_t.long()]
            node_order.append(sigma_t)
            # mask node
            p = self.masker.mask_node(p, sigma_t)
        return node_order

    def uniform_node_decay_ordering(self, datapoint):
        '''
        Samples next node from uniform distribution 
        '''
        p = datapoint.clone()
        return torch.randperm(p.x.shape[0]).tolist()


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

    def preprocess(self, graph):
        '''
        Preprocesses graph to be used by the denoising network.
        '''
        graph = graph.clone()
        graph = self.masker.fully_connect(graph)
        return graph

    def train_step(
            self,
            train_data,
            val_data,
            M = 4, # number of diffusion trajectories to be created for each graph
        ):
        
        self.denoising_optimizer.zero_grad()
        self.ordering_optimizer.zero_grad()

        self.denoising_network.train()
        self.diffusion_ordering_network.eval()
        loss = torch.tensor(0.0, requires_grad=True)
        acc_loss = 0.0
        with tqdm(train_data) as pbar:
            for graph in pbar:
                n_i = graph.x.shape[0]
                # preprocess graph
                graph = self.preprocess(graph)
                diffusion_trajectories = self.generate_diffusion_trajectories(graph, M)  
                # predictions & loss
                for diffusion_trajectory in diffusion_trajectories:
                    G_0 = diffusion_trajectory[0]
                    node_order = self.node_decay_ordering(G_0) # Due to permutation invariance, we can use sample from uniform distribution
                    for t in range(len(node_order)-1, 0, -1):
                        node = node_order[t]
                        G_t = diffusion_trajectory[t].clone()
                        G_tplus1 = diffusion_trajectory[t+1].clone()
                        G_pred = G_tplus1.clone()
                        

                        # predict node type
                        node_type_probs, edge_type_probs = self.denoising_network(G_pred)
                        
                        # sample node type
                        node_type = torch.distributions.Categorical(probs=node_type_probs.squeeze()).sample()

                        # sample edge type
                        new_connections = torch.multinomial(edge_type_probs.squeeze(), num_samples=1, replacement=True)

                       
                        # probability of new connections
                        p_edges = torch.prod(edge_type_probs[new_connections])

                        # calculate loss                     
                        p_O_v =  p_edges*node_type_probs[node_type] + EPSILON
                        w_k = self.diffusion_ordering_network(G_tplus1)[node]
                        loss = (n_i/(len(diffusion_trajectory)-1))*torch.log(p_O_v)*w_k/M # cumulative, to join (k) from all previously denoised nodes
                        acc_loss += loss.item()
                        # backprop (accumulated gradients)
                        loss.backward()
                        pbar.set_description(f"Loss: {loss.item():.4f}")
        
        # update parameters using accumulated gradients
        self.denoising_optimizer.step()
        
        # log loss
        wandb.log({"loss": acc_loss})


        # validation batch (for diffusion ordering network)
        self.denoising_network.eval()
        self.diffusion_ordering_network.train()

        reward = torch.tensor(0.0, requires_grad=True)
        acc_reward = 0.0
        with tqdm(val_data) as pbar:
            for graph in pbar:
                # preprocess graph
                graph = self.preprocess(graph)

                n_i = graph.x.shape[0]
                
                diffusion_trajectories = self.generate_diffusion_trajectories(graph, M)

                for diffusion_trajectory in diffusion_trajectories:
                    G_0 = diffusion_trajectory[0]
                    node_order = self.node_decay_ordering(G_0) # Due to permutation invariance, we can use sample from uniform distribution
                    for t in range(len(node_order)-1, 0, -1):
                        node = node_order[t]
                        G_tplus1 = diffusion_trajectory[t+1]
                        # predict node type
                        node_type_probs, edge_type_probs = self.denoising_network(G_tplus1)

                        # sample node type
                        node_type = torch.distributions.Categorical(probs=node_type_probs.squeeze()).sample()

                        # sample edge type
                        new_connections = torch.multinomial(edge_type_probs.squeeze(), num_samples=1, replacement=True)

                        # probability of new connections
                        p_edges = torch.prod(edge_type_probs[new_connections])

                        # calculate reward (VLB)                        
                        p_O_v =  p_edges*node_type_probs[node_type] + EPSILON

                        r = (n_i/(len(diffusion_trajectory)-1))*torch.log(p_O_v)
                        w_k = self.diffusion_ordering_network(G_tplus1)[node]

                        reward = w_k*r/M
                        acc_reward -= reward.item()
                        pbar.set_description(f"Reward: {reward.item():.4f}")

                        reward.backward()
                        

        
        wandb.log({"reward": acc_reward})
        # update parameters (REINFORCE algorithm)
        self.ordering_optimizer.step()
        


    def save_model(self):
        torch.save(self.denoising_network.state_dict(), "denoising_network_overfit.pt")
        torch.save(self.diffusion_ordering_network.state_dict(), "diffusion_ordering_network_overfit.pt")

    def load_model(self):
        self.denoising_network.load_state_dict(torch.load("denoising_network_overfit.pt"))
        self.diffusion_ordering_network.load_state_dict(torch.load("diffusion_ordering_network_overfit.pt"))