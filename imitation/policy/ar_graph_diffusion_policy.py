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

import logging

log = logging.getLogger(__name__)

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
        self.model = denoising_network
        # no need for diffusion ordering network

        self.masker = NodeMasker(dataset)
        self.global_epoch = 0

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=200, factor=0.5, verbose=True, min_lr=lr/20)
        self.mode = mode

        if ckpt_path is not None:
            self.load_nets(ckpt_path)
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
                    # remove object nodes
                    for obj_node in graph.edge_index.unique()[graph.x[:,0,-1] == self.dataset.OBJECT_NODE_TYPE]:
                        graph = self.masker.remove_node(graph, obj_node)
                    graph = self.masker.idxify(graph)
                    diffusion_trajectory = self.generate_diffusion_trajectory(graph)  
                    # predictions & loss
                    G_0 = diffusion_trajectory[0].to(self.device)
                    node_order = self.node_decay_ordering(G_0.x.shape[0])
                    acc_loss = 0
                    for t in range(len(node_order)):
                        G_pred = diffusion_trajectory[t+1].clone().to(self.device)
                        # calculate joint_poses as edge_attr, using pairwise distance (based on edge_index)
                        joint_values, pos = self.model(G_pred.x,
                                                       G_pred.edge_index,
                                                       G_pred.edge_attr,
                                                       x_coord=G_pred.y[:G_pred.x.shape[0],-1,:3],
                                                       cond=G_0.y[:,:,3:].float())
                        
                        # calculate loss
                        loss = self.loss_fcn(pred_feats=joint_values,
                                             pred_pos=pos,
                                             target_feats=G_0.x[node_order[t],:,:].float(),
                                             target_pos=G_0.pos[:G_pred.x.shape[0],:3].float())
                        acc_loss += loss.item()
                    total_loss += acc_loss
            return total_loss / len(dataset)
        

    def train(self, dataset, num_epochs=100, model_path=None, seed=0):
        '''
        Train noise prediction model
        '''

        # set seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            
        try:
            self.load_nets(model_path)
        except:
            pass
        self.optimizer.zero_grad()
        self.model.train()

        dataloader = torch_geometric.data.DataLoader(dataset, batch_size=1, shuffle=True)

        with tqdm(range(num_epochs), desc='Epoch', leave=False) as tepoch:
            for epoch in tepoch:
                # batch loop
                with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                    for nbatch in tepoch:
                        # preprocess graph
                        graph = self.preprocess(nbatch)
                        # remove object nodes
                        for obj_node in graph.edge_index.unique()[graph.x[:,0,-1] == self.dataset.OBJECT_NODE_TYPE]:
                            graph = self.masker.remove_node(graph, obj_node)
                        graph = self.masker.idxify(graph)
                        diffusion_trajectory = self.generate_diffusion_trajectory(graph)  
                        # predictions & loss
                        G_0 = diffusion_trajectory[0].to(self.device)
                        node_order = self.node_decay_ordering(G_0.x.shape[0])
                        acc_loss = 0
                        
                        # loop over nodes
                        for t in range(len(node_order)):
                            G_pred = diffusion_trajectory[t+1].clone().to(self.device)
                            # calculate joint_poses as edge_attr, using pairwise distance (based on edge_index)
                            joint_values, pos = self.model(G_pred.x, G_pred.edge_index, G_pred.edge_attr, x_coord=G_pred.y[:G_pred.x.shape[0],-1,:3], cond=G_0.y[:,:,3:].float()) # only use quaternions, since pos is x_coord

                            # mse loss for node features
                            loss = self.loss_fcn(pred_feats=joint_values,
                                                 pred_pos=pos,
                                                 target_feats=G_0.x[node_order[t],:,:].float(),
                                                 target_pos=G_0.pos[:G_pred.x.shape[0],:3].float())
                            # TODO add loss for absolute positions, to make the model physics-informed
                            wandb.log({"epoch": self.global_epoch, "loss": loss.item()})

                            acc_loss += loss.item()
                            # backprop (accumulated gradients)
                            loss.backward()
                        # update weights
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.scheduler.step(acc_loss)
                        self.save_nets(model_path)
                        wandb.log({"graph_loss": acc_loss, "learning_rate": self.optimizer.param_groups[0]['lr']})
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
            return x[:9,:,0].T # all (joint-representing) nodes, all timesteps, first value
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
            pos.append(obs_deque[i].pos)
        obs_cond = torch.cat(obs_cond, dim=1)
        obs_pos = torch.cat(pos, dim=0)
        return obs_cond, edge_index, edge_attr, obs_pos

    def MOCK_get_graph_from_obs(self, obs_deque): # for testing purposes, remove before merge
        # plays back observation from dataset
        playback_graph = self.preprocess(self.dataset[self.playback_count])
        playback_graph = self.masker.idxify(playback_graph)
        obs_cond    = playback_graph.y
        edge_index  = playback_graph.edge_index
        edge_attr   = playback_graph.edge_attr
        obs_pos     = playback_graph.y[:,-1,:]
        self.playback_count += 1
        log.debug(f"Playing back observation {self.playback_count}")
        return obs_cond, edge_index, edge_attr, obs_pos


    def pos_from_pos_diffs(self, pos_diffs, edge_index):
        '''
        Calculate absolute positions from position differences between nodes
        pos_diffs: [n_edges, 7]
        edge_index: [2, n_edges]
        '''
        pos = torch.zeros((edge_index.max() + 1, 7))
        for i in range(edge_index.max() + 1):
            pos[i,:3] = pos_diffs[torch.logical_and(edge_index[1,:] == i, edge_index[0,:] == 0),:3]
        return pos


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



        for x_i in range(obs[0].x.shape[0] - 1): # number of nodes in action graph TODO remove objects
            action = self.preprocess(action)
            # predict node attributes for last node in action
            action.x[-1], pos = self.model(
                action.x.float(),
                action.edge_index,
                action.edge_attr,
                x_coord = obs_pos[:action.x.shape[0],:3],
                cond=obs_cond[:,:,3:] # only use quaternions, since pos is x_coord
            )
            action.x[-1,:,-1] = self.dataset.ROBOT_NODE_TYPE # set node type to robot to avoid propagating error
            # map edge attributes from obs to action
            action.edge_attr = self._lookup_edge_attr(edge_index, edge_attr, action.edge_index)
            if x_i == obs[0].x.shape[0] - 1 - 1: # TODO remove objects
                break
            action = self.masker.add_masked_node(action)
            
        joint_values_t = self.get_joint_values(action.x.detach().cpu().numpy())

        return joint_values_t