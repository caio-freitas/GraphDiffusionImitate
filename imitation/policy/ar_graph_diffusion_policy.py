'''
Diffusion Policy for imitation learning with graphs
'''
from functools import lru_cache
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import wandb
import torch_geometric
from imitation.utils.graph_diffusion import NodeMasker

from diffusers.optimization import get_scheduler

import logging

log = logging.getLogger(__name__)

class AutoregressiveGraphDiffusionPolicy(nn.Module):
    def __init__(self,
                 dataset,
                 node_feature_dim,
                 action_dim,
                 num_edge_types,
                 denoising_network,
                 lr=1e-4,
                 ckpt_path=None,
                 device = None,
                 mode = 'joint-space',
                 use_normalization = False,):
        super(AutoregressiveGraphDiffusionPolicy, self).__init__()
        if device == None:
           self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.dataset = dataset
        self.node_feature_dim = node_feature_dim
        self.action_dim = action_dim
        self.num_edge_types = num_edge_types
        self.model = denoising_network
        self.use_normalization = use_normalization
        self.masker = NodeMasker(dataset)
        self.global_epoch = 0

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                            lr=lr,
                                            weight_decay=1e-6,
                                            betas=[0.95, 0.999],
                                            eps=1e-8)
        self.mode = mode

        self.lr_scheduler = None
        self.num_epochs = None
        
        self.playback_count = 0 # for testing purposes, remove before merge

    def load_nets(self, ckpt_path):
        '''
        Load networks from checkpoint
        '''
        print(f"Loading networks from checkpoint: {ckpt_path}")
        try:
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        except Exception as e:
            print(f"Could not load networks from checkpoint: {e}\n")


    def save_nets(self, ckpt_path):
        '''
        Save networks to checkpoint
        '''
        torch.save(self.model.state_dict(), ckpt_path)

    @torch.jit.export
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

    @torch.jit.export
    def preprocess(self, graph):
        '''
        Preprocesses graph to be used by the denoising network.
        '''
        graph = graph.clone()
        graph = self.masker.fully_connect(graph)
        graph.x = graph.x.float()
        graph.edge_attr = graph.edge_attr.long()
        return graph

    # cache function results, as it is called multiple times
    @lru_cache
    def node_decay_ordering(self, graph_size):
        '''
        Returns node decay ordering
        '''
        return torch.arange(graph_size-1, -1, -1)

    def loss_fcn(self, pred_feats, pred_pos, target_feats, target_pos):
        '''
        Node feature loss
        '''
        lambda_joint_pos = 0.1
        lambda_joint_values = 1

        pred_joint_vals = pred_feats[:,0] # [pred_horizon, 1]
        target_joint_vals = target_feats[:,0] # [pred_horizon, 1]

        loss_joint_pos_loss = F.pairwise_distance(pred_pos, target_pos, p=2).mean()
        wandb.log({"loss_joint_pos_loss": loss_joint_pos_loss.item()})
        loss_joint_values = nn.MSELoss()(pred_joint_vals, target_joint_vals)
        wandb.log({"loss_joint_values": loss_joint_values.item()})

        return lambda_joint_values * loss_joint_values # + lambda_joint_pos * loss_joint_pos


    def validate(self, dataset, model_path=None):
        '''
        Calculate validation loss for noise prediction model in the given dataset
        '''
        self.load_nets(model_path)
        self.model.eval()

        with torch.no_grad():
            total_loss = 0
            with tqdm(dataset, desc='Val Batch', leave=False) as tbatch:
                for nbatch in tbatch:
                    graph = self.preprocess(nbatch)
                    if self.use_normalization:
                        graph.x = self.dataset.normalize_data(graph.x, stats_key="action")
                        graph.y = self.dataset.normalize_data(graph.y, stats_key="obs")
                    
                    graph = self.masker.idxify(graph)
                    # FiLM generator
                    embed = self.model.cond_encoder(graph.y, graph.edge_index, graph.pos[:,:3], graph.edge_attr.unsqueeze(-1))
                    # remove object nodes
                    for obj_node in graph.edge_index.unique()[graph.x[:,0,-1] == self.dataset.OBJECT_NODE_TYPE]:
                        graph = self.masker.remove_node(graph, obj_node)
                    diffusion_trajectory = self.generate_diffusion_trajectory(graph) 
                    dataloader = torch_geometric.data.DataLoader(diffusion_trajectory, 
                                                                     batch_size=len(diffusion_trajectory), 
                                                                     shuffle=False)
                    G_pred = next(iter(dataloader))
                    # predictions & loss
                    G_0 = graph.to(self.device)
                    node_order = self.node_decay_ordering(G_0.x.shape[0])
                    G_pred = diffusion_trajectory[t+1].clone().to(self.device)
                    # calculate joint_poses as edge_attr, using pairwise distance (based on edge_index)
                    joint_values, pos = self.model(G_pred.x[:,:,:self.node_feature_dim],
                                                    G_pred.edge_index,
                                                    G_pred.edge_attr,
                                                    x_coord=G_pred.pos[:,:3],
                                                    film_cond=embed,
                                                    batch=G_pred.batch)
                    
                    joint_values = joint_values[G_pred.ptr[1:]-1]  # node being masked, this assumes that the masked node is the last node in the graph
                    pos = pos[G_pred.ptr[1:]-1]

                    # calculate loss
                    loss = self.loss_fcn(pred_feats=joint_values,
                                            pred_pos=pos,
                                            target_feats=G_0.x.float(),
                                            target_pos=G_0.pos[:G_pred.x.shape[0],:3].float())
                    total_loss += loss.item()
            return total_loss / len(dataset)
        

    def train(self, dataset, num_epochs=100, model_path=None, seed=0):
        '''
        Train noise prediction model
        '''

        # set seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        if self.num_epochs is None:
            log.warn(f"Global num_epochs not set. Using {num_epochs}.")
            self.num_epochs = num_epochs
        try:
            self.load_nets(model_path)
        except:
            pass
        self.optimizer.zero_grad()
        self.model.train()

        if self.lr_scheduler is None:
        # Cosine LR schedule with linear warmup
            self.lr_scheduler = get_scheduler(
                name='cosine',
                optimizer=self.optimizer,
                num_warmup_steps=50,
                num_training_steps=len(dataset) * self.num_epochs
            )

        with tqdm(range(num_epochs), desc='Epoch', leave=False) as tepoch:
            for epoch in tepoch:
                # batch loop
                with tqdm(dataset, desc='Batch', leave=False) as tepoch:
                    for nbatch in tepoch:
                        # preprocess graph
                        graph = self.preprocess(nbatch)
                        if self.use_normalization:
                            graph.x = self.dataset.normalize_data(graph.x, stats_key="action")
                            graph.y = self.dataset.normalize_data(graph.y, stats_key="obs")
                       
                        graph = self.masker.idxify(graph)
                        # FiLM generator
                        embed = self.model.cond_encoder(graph.y, graph.edge_index, graph.pos[:,:3], graph.edge_attr.unsqueeze(-1))
                        # remove object nodes. Last nodes first, to avoid index errors
                        for obj_node in torch.flip(graph.edge_index.unique()[graph.x[:,0,-1] == self.dataset.OBJECT_NODE_TYPE], dims=[0]):
                            graph = self.masker.remove_node(graph, obj_node)
                        diffusion_trajectory = self.generate_diffusion_trajectory(graph)  
                        dataloader = torch_geometric.data.DataLoader(diffusion_trajectory, 
                                                                     batch_size=len(diffusion_trajectory), 
                                                                     shuffle=False)
                        G_pred = next(iter(dataloader))
                        # predictions & loss
                        G_0 = diffusion_trajectory[0].to(self.device)
                        node_order = self.node_decay_ordering(G_0.x.shape[0])
                        
                        # loop over nodes
                        # calculate joint_poses as edge_attr, using pairwise distance (based on edge_index)
                        joint_values, pos = self.model(G_pred.x[:,:,:self.node_feature_dim],
                                                        G_pred.edge_index,
                                                        G_pred.edge_attr,
                                                        x_coord=G_pred.pos[:,:3],
                                                        film_cond=embed,
                                                        batch=G_pred.batch) # only use node type, E(3) invariant
                        
                        joint_values = joint_values[G_pred.ptr[1:]-1]  # node being masked, this assumes that the masked node is the last node in the graph
                        pos = pos[G_pred.ptr[1:]-1]

                        # mse loss for node features
                        loss = self.loss_fcn(pred_feats=joint_values,
                                                pred_pos=pos,
                                                target_feats=G_0.x[:,:,:self.node_feature_dim].float(),
                                                target_pos=G_0.pos[:,:3].float())
                        # TODO add loss for absolute positions, to make the model physics-informed
                        wandb.log({"epoch": self.global_epoch, "loss": loss.item()})

                        acc_loss = loss.item()
                        # backprop (accumulated gradients)
                        loss.backward(retain_graph=True)
                        # update weights
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.lr_scheduler.step()
                        self.save_nets(model_path)
                        wandb.log({"graph_loss": acc_loss, "lr": self.lr_scheduler.get_last_lr()[0]})
                self.global_epoch += 1

    def get_joint_values(self, x):
        '''
        Return joint value commands from node feature vector
        Depends on operation mode:
        - joint-space: return first element
        - task-joint-space: return first element
        - end-effector: raise NotImplementedError
        '''
        if self.mode == 'joint-space' or self.mode == 'task-joint-space':
            return x[:self.action_dim,:,0].T # all (joint-representing) nodes, all timesteps, first value
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
            action_edge_attr[i] = edge_attr[torch.logical_and(edge_index[0] == action_edge_index[0, i],
                                                              edge_index[1] == action_edge_index[1, i])]
        return action_edge_attr
    
    def get_graph_from_obs(self, obs_deque):
        '''
        Get graph from observation deque
        '''
        obs_cond = []
        pos = []
        # edge_indes and edge_attr don't change
        first_graph = self.preprocess(obs_deque[0])
        first_graph = self.masker.idxify(first_graph)
        edge_index = first_graph.edge_index # edge_index doesn't change over time
        edge_attr = first_graph.edge_attr # edge_attr doesn't change over time
        for i in range(len(obs_deque)):
            obs_cond.append(obs_deque[i].y.unsqueeze(1))
        obs_cond = torch.cat(obs_cond, dim=1)
        if self.use_normalization:
            obs_cond = self.dataset.normalize_data(obs_cond, stats_key="obs")
        return obs_cond, edge_index, edge_attr, obs_deque[-1].pos

    def MOCK_get_graph_from_obs(self, obs_deque): # for testing purposes, remove before merge
        # plays back observation from dataset
        playback_graph = self.preprocess(self.dataset[self.playback_count])
        playback_graph = self.masker.idxify(playback_graph)
        obs_cond    = playback_graph.y
        edge_index  = playback_graph.edge_index
        edge_attr   = playback_graph.edge_attr
        obs_pos     = playback_graph.pos
        self.playback_count += 1
        log.debug(f"Playing back observation {self.playback_count}")
        return obs_cond, edge_index, edge_attr, obs_pos


    def get_action(self, obs):
        '''
        Get action from observation
        obs: deque of observations from lowdim runner
        '''
        # append x, edge_index, edge_attr from all graphs in obs to single graph
        assert len(obs) == self.dataset.obs_horizon
        obs_cond, edge_index, edge_attr, obs_pos = self.get_graph_from_obs(obs)
       
        self.model.eval()

        # graph action representation: x, edge_index, edge_attr
        action = self.masker.create_empty_graph(1) # one masked node
        # FiLM generator
        embed = self.model.cond_encoder(obs_cond, edge_index, obs_pos[:,:3], edge_attr.unsqueeze(-1))

        if self.use_normalization:
            obs_cond = self.dataset.normalize_data(obs_cond, stats_key="obs")

        for x_i in range(obs[0].x.shape[0] - 1): # number of nodes in action graph TODO remove objects
            action = self.preprocess(action)
            # predict node attributes for last node in action
            joint_feats, pos = self.model(
                action.x[:,:,:self.node_feature_dim].float(),
                action.edge_index,
                action.edge_attr,
                x_coord = obs_pos[:action.x.shape[0],:3],
                film_cond=embed)
            action.x[-1,:,:self.node_feature_dim] = joint_feats[-1,...] # attribute prediction to last node
            action.x[-1,:,-1] = self.dataset.ROBOT_NODE_TYPE # set node type to robot to avoid propagating error
            # map edge attributes from obs to action
            action.edge_attr = self._lookup_edge_attr(edge_index, edge_attr, action.edge_index)
            if x_i == obs[0].x.shape[0] - 1:
                break
            action = self.masker.add_masked_node(action)
            
        if self.use_normalization:
            action.x = self.dataset.unnormalize_data(action.x, stats_key="action")
        joint_values_t = self.get_joint_values(action.x.detach().cpu().numpy())

        return joint_values_t