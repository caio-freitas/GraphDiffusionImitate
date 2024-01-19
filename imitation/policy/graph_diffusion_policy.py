'''
Diffusion Policy for imitation learning with graphs
'''
import torch
from tqdm import tqdm
import torch.nn as nn
import wandb
import torch_geometric
from imitation.model.graph_diffusion import  DenoisingNetwork
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
                 device = None):
        super(GraphDiffusionPolicy, self).__init__()
        if device == None:
           self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.dataset = dataset
        self.node_feature_dim = node_feature_dim
        self.num_edge_types = num_edge_types
        self.denoising_network = denoising_network
        # no need for diffusion ordering network
        self.node_feature_loss = nn.MSELoss()
        self.masker = NodeMasker(dataset)
        

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

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

    def load_nets(self, ckpt_path):
        '''
        Load networks from checkpoint
        '''
        self.denoising_network.load_state_dict(ckpt_path)


    def save_nets(self, ckpt_path):
        '''
        Save networks to checkpoint
        '''
        torch.save(self.denoising_network.state_dict(), ckpt_path)

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
        graph.x = graph.x.float()
        graph.edge_attr = graph.edge_attr.long()
        return graph

    def node_decay_ordering(self, graph):
        '''
        Returns node decay ordering
        '''
        return torch.arange(graph.x.shape[0]-1, -1, -1)

    def vlb(self, G_0, edge_type_probs, node, node_order, t):
        T = len(node_order)
        n_i = G_0.x.shape[0]
        # retrieve edge type from G_t.edge_attr, edges between node and node_order[t:]
        edge_attrs_matrix = G_0.edge_attr.reshape(T, T)
        original_edge_types = torch.index_select(edge_attrs_matrix[node], 0, torch.tensor(node_order[t:]).to(self.device))
        # calculate probability of edge type
        p_edges = torch.gather(edge_type_probs, 1, original_edge_types.reshape(-1, 1))
        log_p_edges = torch.sum(torch.log(p_edges))
        # log_p_edges = torch.sum(torch.tensor([0]))
        wandb.log({"target_edges_log_prob": log_p_edges})
        # calculate loss
        log_p_O_v =  log_p_edges
        loss = -(n_i/T)*log_p_O_v # cumulative, to join (k) from all previously denoised nodes
        return loss

    def train(self, dataset, num_epochs=100, model_path=None, seed=0):
        '''
        Train noise prediction model
        '''
        self.optimizer.zero_grad()
        self.denoising_network.train()

        with tqdm(range(num_epochs), desc='Epoch', leave=False) as tepoch:
            for epoch in tepoch:
                # batch loop
                with tqdm(dataset, desc='Batch', leave=False) as tepoch:
                    for nbatch in tepoch:
                        # preprocess graph
                        graph = self.preprocess(nbatch)
                        diffusion_trajectory = self.generate_diffusion_trajectory(graph)  
                        # predictions & loss
                        G_0 = diffusion_trajectory[0].to(self.device)
                        node_order = self.node_decay_ordering(G_0)
                        acc_loss = 0
                        self.optimizer.zero_grad()
                        
                        # generate initial node embeddings
                        _ ,_ , h_v = self.denoising_network(G_0.x.float(), G_0.edge_index, G_0.edge_attr.float())
                        # loop over nodes
                        for t in range(len(node_order)):
                            G_pred = diffusion_trajectory[t+1].clone().to(self.device)                      

                            # predict node and edge type distributions
                            node_features, edge_type_probs, h_v = self.denoising_network(G_pred.x.float(), G_pred.edge_index, G_pred.edge_attr.float(), h_v = h_v[:G_pred.x.shape[0], :])

                            # calculate loss relative to edge type distribution
                            # loss = self.vlb(G_0, edge_type_probs, node_order[t], node_order, t) # cumulative loss
                            # mse loss for node features
                            loss = self.node_feature_loss(node_features[node_order[t]], G_0.x[node_order[t]])
                            # wandb.log({"vlb": loss.item(), "node_feature_loss": node_loss.item()})
                            # loss += node_loss
                            wandb.log({"loss": loss.item()})

                            acc_loss += loss.item()
                            # backprop (accumulated gradients)
                            loss.backward(retain_graph=True)
                        self.optimizer.step()
                        self.save_nets(model_path)
                        wandb.log({"epoch_loss": acc_loss})


    def get_action(self, obs):
        '''
        Get action from observation
        '''
        # append x, edge_index, edge_attr from all graphs in obs to single graph
        assert len(obs) == self.dataset.obs_horizon
        node_features = []
        edge_index = []
        edge_attr = []
        for i in range(len(obs)):
            node_features.append(obs[i].x)
            edge_index.append(obs[i].edge_index)
            edge_attr.append(obs[i].edge_attr)
        node_features = torch.cat(node_features, dim=1)
        edge_index = torch.cat(edge_index, dim=1)
        edge_attr = torch.cat(edge_attr, dim=0)
        obs = torch_geometric.data.Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

        # preprocess obs
        obs = self.preprocess(obs)
        self.denoising_network.eval()


        # predict node and edge type distributions
        node_features, edge_type_probs, h_v = self.denoising_network(obs.x.float(), obs.edge_index, obs.edge_attr.float())
        obs.x = node_features.detach().cpu().numpy()
        
        return obs.x # return joint positions only
