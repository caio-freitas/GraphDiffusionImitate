'''
Diffusion Policy for imitation learning with graphs
'''
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import wandb
import torch_geometric
from imitation.model.graph_diffusion import  DenoisingNetwork
from imitation.utils.graph_diffusion import NodeMasker
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler


class AutoregressiveGraphDiffusionPolicy(nn.Module):
    def __init__(self,
                 dataset,
                 node_feature_dim,
                 num_edge_types,
                 denoising_network,
                 lr=1e-4,
                 ckpt_path=None,
                 device = None,
                 mode = 'joint-space'):
        super(AutoregressiveGraphDiffusionPolicy, self).__init__()
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
        

        self.optimizer = torch.optim.AdamW(self.denoising_network.parameters(), lr=5e-4)
        self.mode = mode

        if ckpt_path is not None:
            self.load_nets(ckpt_path)

    def load_nets(self, ckpt_path):
        '''
        Load networks from checkpoint
        '''
        print(f"Loading networks from checkpoint: {ckpt_path}")
        try:
            self.denoising_network.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        except Exception as e:
            print(f"Could not load networks from checkpoint: {e}\n")


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
        try:
            self.load_nets(model_path)
        except:
            pass
        self.optimizer.zero_grad()
        self.denoising_network.train()
        batch_size = 5

        with tqdm(range(num_epochs), desc='Epoch', leave=False) as tepoch:
            for epoch in tepoch:
                batch_i = 0
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
                        
                        # loop over nodes
                        for t in range(len(node_order)):
                            G_pred = diffusion_trajectory[t+1].clone().to(self.device)

                            # predict node and edge type distributions
                            node_features = self.denoising_network(G_pred.x.float(), G_pred.edge_index, G_pred.edge_attr.float(), cond=G_0.y.float())

                            # mse loss for node features
                            loss = self.node_feature_loss(node_features, G_0.x[node_order[t],:,0].float())
                            # use correlation as loss
                            # x = torch.stack([node_features.squeeze(), G_0.x[node_order[t],:,0].float()])
                            # x += torch.rand_like(x) * 1e-8 # add noise to avoid NaNs
                            # loss -= (1/batch_size)*torch.corrcoef(x)[0,1]
                            wandb.log({"loss": loss.item()})

                            acc_loss += loss.item()
                            # backprop (accumulated gradients)
                            loss.backward(retain_graph=True)
                        batch_i += 1
                        # update weights
                        if batch_i % batch_size == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            self.save_nets(model_path)
                            wandb.log({"batch_loss": acc_loss})

    def get_joint_values(self, x):
        '''
        Return joint value commands from node feature vector
        Depends on operation mode:
        - joint-space: return first element
        - task-joint-space: return first element
        - end-effector: raise NotImplementedError
        '''
        if self.mode == 'joint-space' or self.mode == 'task-joint-space':
            return x[:,:,0].T # all nodes, all timesteps, first value
        elif self.mode == 'end-effector':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _lookup_edge_attr(self, edge_index, edge_attr, action_edge_index):
        '''
        Lookup edge attributes from obs to action
        '''
        # edge_attr = edge_attr.float()
        action_edge_attr = torch.zeros(action_edge_index.shape[1])
        for i in range(action_edge_index.shape[1]):
            # find edge in obs
            action_edge_attr[i] = edge_attr[torch.logical_and(edge_index[0] == action_edge_index[0, i], edge_index[1] == action_edge_index[1, i])]
        return action_edge_attr
    
    def get_action(self, obs):
        '''
        Get action from observation
        obs: deque of observations from lowdim runner
        '''
        # append x, edge_index, edge_attr from all graphs in obs to single graph
        assert len(obs) == self.dataset.obs_horizon
        # turn deque into single graph with extra node-feature dimension
        node_features = []
        # edge_indes and edge_attr don't change
        obs[0] = self.preprocess(obs[0])
        obs[0] = self.masker.idxify(obs[0])
        edge_index = obs[0].edge_index
        edge_attr = obs[0].edge_attr.float()
        for i in range(len(obs)):
            node_features.append(obs[i].x.unsqueeze(1))
        node_features = torch.cat(node_features, dim=1)

       
        self.denoising_network.eval()

        # graph action representation: x, edge_index, edge_attr
        action = self.masker.create_empty_graph(1) # one masked node
        action = self.masker.idxify(action)
        for x_i in range(obs[0].x.shape[0]): # number of nodes in action graph
            # preprocess action graph
            action = self.preprocess(action)    
            # map edge attributes from obs to action
            action.edge_attr = self._lookup_edge_attr(edge_index, edge_attr, action.edge_index)
            # predict node and edge type distributions
            action.x[-1] = self.denoising_network(action.x.float(), action.edge_index, action.edge_attr, cond=node_features)
            if x_i == obs[0].x.shape[0]-1:
                break
            action = self.masker.add_masked_node(action)
            
            
        joint_values_t = self.get_joint_values(action.x.detach().cpu().numpy())

        print("joint values: ", joint_values_t)
        return joint_values_t