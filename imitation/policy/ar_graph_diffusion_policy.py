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
from torch_geometric.loader import DataLoader
from imitation.utils.graph_diffusion import NodeMasker


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
        Initially permutation variant, since node ordering is known (robot nodes first, then object node(s))
        '''
        diffusion_trajectory = []
        target_node_features = [] # node features that get masked, in the order they are masked

        node_order = self.node_decay_ordering(graph.x.shape[0])

        masked_data = graph.clone()
        for t in range(len(node_order)):
            node = node_order[t]
            masked_data = masked_data.clone()
            target_node_features.append(masked_data.x[node])
            masked_data = self.masker.mask_node(masked_data, node)
            masked_data.x = masked_data.x.float().to(self.device)
            masked_data.edge_attr = masked_data.edge_attr.long().to(self.device)
            masked_data.edge_index = masked_data.edge_index.long().to(self.device)
            masked_data.y = masked_data.y.float().to(self.device)
            masked_data.pos = masked_data.pos.float().to(self.device)
            diffusion_trajectory.append(masked_data)
            # don't remove last node
            if t < len(node_order)-1:
                masked_data = self.masker.remove_node(masked_data, node)
                node_order = [n-1 if n > node else n for n in node_order] # update node order to account for removed node
        return diffusion_trajectory, target_node_features

    @torch.jit.export
    def preprocess(self, graph):
        '''
        Preprocesses graph to be used by the denoising network.
        '''
        graph = graph.clone()
        graph = self.masker.fully_connect(graph)
        graph.x = graph.x.float().to(self.device)
        graph.edge_attr = graph.edge_attr.long().to(self.device)
        graph.edge_index = graph.edge_index.long().to(self.device)
        # graph.y = graph.y.float().to(self.device)
        # graph.pos = graph.pos.float().to(self.device)
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

        pred_joint_vals = pred_feats[:,:,0] # [pred_horizon, 1]
        target_joint_vals = target_feats[:,:,0] # [pred_horizon, 1]
        loss_joint_pos_loss = F.pairwise_distance(pred_pos, target_pos, p=2).mean()
        wandb.log({"loss_joint_pos_loss": loss_joint_pos_loss.item()})
        loss_joint_values = nn.MSELoss()(pred_joint_vals, target_joint_vals)
        wandb.log({"loss_joint_values": loss_joint_values.item()})

        return lambda_joint_values * loss_joint_values + lambda_joint_pos * loss_joint_pos_loss



    def train(self, dataset, num_epochs=100, model_path=None, seed=0):
        '''
        Train noise prediction model
        '''
        try:
            self.load_nets(model_path)
        except:
            pass
        self.optimizer.zero_grad()
        self.model.train()
        batch_size = 5

        with tqdm(range(num_epochs), desc='Epoch', leave=False) as tepoch:
            for epoch in tepoch:
                batch_i = 0
                # batch loop
                with tqdm(dataset, desc='Batch', leave=False) as tepoch:
                    for nbatch in tepoch:
                        # preprocess graph
                        graph = self.preprocess(nbatch)
                        # remove object nodes
                        graph = self.masker.remove_node(graph, 9) # TODO properly remove object nodes
                        graph = self.masker.idxify(graph)
                        diffusion_trajectory, target_node_features = self.generate_diffusion_trajectory(graph)
                        target_node_features = torch.stack(target_node_features, dim=0) # [n_nodes, pred_horizon, n_features]
                        dataloader = DataLoader(dataset=diffusion_trajectory, batch_size=len(diffusion_trajectory), shuffle=False)
                        G_pred = next(iter(dataloader)).to(self.device)
                        # predictions & loss
                        G_0 = diffusion_trajectory[0].to(self.device)
                        acc_loss = 0
                        # calculate joint_poses as edge_attr, using pairwise distance (based on edge_index)
                        joint_values, pos = self.model(G_pred, x_coord=G_pred.y[:,-1,:3], cond=G_0.y[:,:,:3].float())
                        # get elements from G_pred.ptr
                        joint_values = joint_values[G_pred.ptr[1:] - 1]
                        # mse loss for node features
                        loss = self.loss_fcn(pred_feats=joint_values,
                                                pred_pos=pos,
                                                target_feats=target_node_features.float(),
                                                target_pos=G_pred.y[:,-1,:3].float())
                        # TODO add loss for absolute positions, to make the model physics-informed
                        wandb.log({"epoch": self.global_epoch, "loss": loss.item()})

                        acc_loss += loss.item()
                        # backprop (accumulated gradients)
                        loss.backward(retain_graph=True)
                        batch_i += 1
                        # update weights
                        if batch_i % batch_size == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            self.scheduler.step(acc_loss)
                            self.save_nets(model_path)
                            wandb.log({"batch_loss": acc_loss, "learning_rate": self.optimizer.param_groups[0]['lr']})
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
            return x[:8,:,0].T # all (joint-representing) nodes, all timesteps, first value
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
            obs_cond.append(obs_deque[i].y[:,:3]) # only positions
            pos.append(obs_deque[i].pos)
        obs_cond = torch.stack(obs_cond, dim=1)
        obs_pos = torch.stack(pos, dim=1)
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
        # TODO add asserts for node feature dimensio, edge feature dimensions, poses, etc.
        obs_cond, edge_index, edge_attr, obs_pos = self.get_graph_from_obs(obs)

        assert obs_pos.shape[0] == obs[0].x.shape[0]
        assert obs_pos.shape[1] == self.dataset.obs_horizon
        assert obs_cond.shape[0] == obs[0].y.shape[0]
        assert obs_cond.shape[1] == self.dataset.obs_horizon
        assert edge_attr.shape[0] == edge_index.shape[1]
       
        self.model.eval()

        # graph action representation: x, edge_index, edge_attr
        action = self.masker.create_empty_graph(1) # one masked node

        pos = torch.zeros((1,3))

        for x_i in range(obs[0].x.shape[0]): # number of nodes in action graph TODO remove objects
            action = self.preprocess(action)
            # predict node attributes for last node in action
            action_pred, pos = self.model(action, x_coord = obs_pos[:x_i+1,-1,:3], cond=obs_cond[:,:,:3])
            action.x[-1,:,:] = action_pred[-1,:,:]
            action.x[-1,:,-1] = self.dataset.ROBOT_NODE_TYPE # set node type to robot to avoid propagating error
            # map edge attributes from obs to action
            action.edge_attr = self._lookup_edge_attr(edge_index, edge_attr, action.edge_index)
            if x_i == obs[0].x.shape[0]-1:
                break
            action.x = action.x.detach().cpu()
            action.edge_attr = action.edge_attr.detach().cpu()
            action.edge_index = action.edge_index.detach().cpu()
            action = self.masker.add_masked_node(action)
            
            
            
        joint_values_t = self.get_joint_values(action.x.detach().cpu().numpy())

        return joint_values_t