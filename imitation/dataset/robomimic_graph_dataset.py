import collections
from typing import Callable, Optional
from torch_geometric.data import Dataset, Data, InMemoryDataset
import logging
import h5py
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict
from functools import lru_cache
from scipy.spatial.transform import Rotation as R

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from imitation.utils.generic import calculate_panda_joints_positions

log = logging.getLogger(__name__)

class RobomimicGraphDataset(InMemoryDataset):
    def __init__(self, 
                 dataset_path,
                 robots,
                 object_state_sizes,
                 object_state_keys,
                 pred_horizon=1,
                 obs_horizon=1,
                 node_feature_dim = 2, # joint value and node type flag
                 control_mode="JOINT_VELOCITY",
                 base_link_shift=[0.0, 0.0, 0.0],
                 base_link_rotation=[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]):
        self.control_mode           : str = control_mode
        self.node_feature_dim       : int = node_feature_dim
        self.obs_feature_dim        : int = 7 # 6 for rotation, 1 for node ID
        self.robots                 : List = robots
        self.num_robots             : int = len(self.robots)
        self.pred_horizon           : int = pred_horizon
        self.obs_horizon            : int = obs_horizon
        self.object_state_sizes     : Dict = object_state_sizes # can be taken from https://github.com/ARISE-Initiative/robosuite/tree/master/robosuite/environments/manipulation
        self.object_state_keys      : Dict = object_state_keys
        self.num_objects            : int = len(object_state_keys)
        self._processed_dir         : str = dataset_path.replace(".hdf5", f"_{self.control_mode}_processed_{self.obs_horizon}_{self.pred_horizon}")

        self.BASE_LINK_SHIFT        : List = base_link_shift
        self.BASE_LINK_ROTATION     : List = base_link_rotation
        self.ROBOT_NODE_TYPE        : int = 1
        self.OBJECT_NODE_TYPE       : int = -1

        self.ROBOT_LINK_EDGE        : int = 1
        self.OBJECT_ROBOT_EDGE      : int = 2

        self.dataset_root           : str = h5py.File(dataset_path, 'r')
        self.dataset_keys           : List = list(self.dataset_root["data"].keys())
        try:
            self.dataset_keys.remove("mask")
        except:
            pass
        self.rotation_transformer = RotationTransformer(
            from_rep="quaternion",
            to_rep="rotation_6d"
        )
        self.eef_idx = [-1, 8] # end-effector index
        if self.num_robots == 2:
            self.eef_idx += [17]

        super().__init__(root=self._processed_dir, transform=None, pre_transform=None, pre_filter=None, log=True)

        self.normalizer = self.get_normalizer()
        

    @property
    def processed_file_names(self):
        '''
        List of files in the self.processed_dir directory that need to be found in order to skip processing
        '''
        names = [f"data_{i}.pt" for i in range(self.len())]
        return names

    @lru_cache(maxsize=None)
    def _get_object_feats(self, num_objects, node_feature_dim, T): # no associated joint value
        # create tensor of same dimension return super()._get_node_feats(data, t) as node_feats
        obj_state_tensor = torch.zeros((num_objects, T, node_feature_dim))
        return obj_state_tensor

    def _get_object_pos(self, data, t):
        obj_state_tensor = torch.zeros((self.num_objects, 9)) # 3 for position, 6 for 6D rotation

        for object, object_state_items in enumerate(self.object_state_keys.values()):
            i = 0
            for object_state in object_state_items:
                if "quat" in object_state:
                    assert self.object_state_sizes[object_state] == 4, "Quaternion must have size 4"
                    rot = self.rotation_transformer.forward(torch.tensor(data["object"][t][i:i + self.object_state_sizes[object_state]]))
                    obj_state_tensor[object,i:i + 6] = rot
                else:
                    obj_state_tensor[object,i:i + self.object_state_sizes[object_state]] = torch.from_numpy(data["object"][t][i:i + self.object_state_sizes[object_state]])
                i += self.object_state_sizes[object_state]

        return obj_state_tensor

    def _get_node_pos(self, data, t):
        node_pos = []
        for i in range(self.num_robots):
            node_pos_robot = calculate_panda_joints_positions([*data[f"robot{i}_joint_pos"][t], *data[f"robot{i}_gripper_qpos"][t]])
            # rotate robot nodes
            rotation_matrix = R.from_quat(self.BASE_LINK_ROTATION[i])
            node_pos_robot[:,:3] = torch.matmul(node_pos_robot[:,:3], torch.tensor(rotation_matrix.as_matrix()))
            node_pos_robot[:,3:] = torch.tensor((R.from_quat(node_pos_robot[:,3:].detach().numpy()) * rotation_matrix).as_quat())
            # add base link shift
            node_pos_robot[:,:3] += torch.tensor(self.BASE_LINK_SHIFT[i])
            node_pos.append(node_pos_robot)
        node_pos = torch.cat(node_pos, dim=0)
        # use rotation transformer to convert quaternion to 6d rotation
        node_pos = torch.cat([node_pos[:,:3], self.rotation_transformer.forward(node_pos[:,3:])], dim=1)
        obj_pos_tensor = self._get_object_pos(data, t)
        node_pos = torch.cat((node_pos, obj_pos_tensor), dim=0)
        return node_pos
    
    def _get_node_feats(self, data, t_vals, control_mode=None):
        '''
        Calculate node features for time steps t_vals
        t_vals: list of time steps
        '''
        T = len(t_vals)
        node_feats = []
        if control_mode is None:
            control_mode = self.control_mode
        if control_mode == "OSC_POSE":
            for i in range(self.num_robots):
                node_feats.append(torch.cat([torch.tensor(data["robot0_eef_pos"][t_vals]), torch.tensor(data["robot0_eef_quat"][t_vals])], dim=0))
        elif control_mode == "JOINT_POSITION":
            for i in range(self.num_robots):
                node_feats.append(torch.cat([
                                torch.tensor(data[f"robot{i}_joint_pos"][t_vals]),
                                torch.tensor(data[f"robot{i}_gripper_qpos"][t_vals])], dim=1).T.unsqueeze(2))
        elif control_mode == "JOINT_VELOCITY":
            for i in range(self.num_robots):
                node_feats.append(torch.cat([
                                torch.tensor(data[f"robot{i}_joint_vel"][t_vals]),
                                torch.tensor(data[f"robot{i}_gripper_qvel"][t_vals])], dim=1).T.unsqueeze(2))
        node_feats = torch.cat(node_feats, dim=0) # [num_robots*num_joints, T, 1]
        obj_state_tensor = self._get_object_feats(self.num_objects, self.node_feature_dim, T)

        # add dimension for NODE_TYPE
        node_feats = torch.cat((node_feats, self.ROBOT_NODE_TYPE*torch.ones((node_feats.shape[0], node_feats.shape[1],1))), dim=2)
        obj_state_tensor[:, :, -1] = self.OBJECT_NODE_TYPE
    
        node_feats = torch.cat((node_feats, obj_state_tensor), dim=0)        
        return node_feats
    
    def get_y_feats(self, data, t_vals):
        '''
        Calculate observation node features for time steps t_vals
        '''
        T = len(t_vals)
        y = []
        for i in range(self.num_robots):
            y.append(torch.cat([
                            torch.tensor(data[f"robot{i}_joint_pos"][t_vals]),
                            torch.tensor(data[f"robot{i}_gripper_qpos"][t_vals])], dim=1).T.unsqueeze(2))
        y = torch.cat(y, dim=0) # [num_robots*num_joints, T, 1]

        obj_state_tensor = []
        for t in t_vals:
            obj_state_tensor.append(self._get_object_pos(data, t))

        obj_state_tensor = torch.stack(obj_state_tensor, dim=1)
        # remove positions
        obj_state_tensor = obj_state_tensor[:,:,3:]

        # add dimensions to match with obj_state_tensor for concatenation
        y = torch.cat((y, torch.zeros((y.shape[0], obj_state_tensor.shape[1], obj_state_tensor.shape[2] - 1))), dim=2)
        
        y = torch.cat((y, obj_state_tensor), dim=0)

        # Add node ID to node features
        node_id = torch.arange(y.shape[0]).unsqueeze(1).unsqueeze(2).repeat(1, T, 1)
        y = torch.cat((y, node_id), dim=2)

        return y

    def _get_node_feats_horizon(self, data, idx, horizon):
        '''
        Calculate node features for self.obs_horizon time steps
        '''
        node_feats = []
        # calculate node features for timesteps idx to idx + horizon
        t_vals = list(range(idx, idx + horizon))
        node_feats = self._get_node_feats(data, t_vals)
        return node_feats
    
    @lru_cache(maxsize=None)
    def _get_edge_attrs(self, edge_index):
        '''
        Attribute edge types to edges
        - self.ROBOT_LINK_EDGE for edges between robot nodes
        - self.OBJECT_ROBOT_EDGE for edges between robot and object nodes
        '''
        edge_attrs = []
        num_nodes = torch.max(edge_index)
        for edge in edge_index.t():
            # num nodes - self.num_objects is the index of the last robot node
            if edge[0] <= num_nodes - self.num_objects and edge[1] <= num_nodes - self.num_objects:
                edge_attrs.append(self.ROBOT_LINK_EDGE)
            # there are no object-to-object edges
            else:
                edge_attrs.append(self.OBJECT_ROBOT_EDGE)
        return torch.tensor(edge_attrs, dtype=torch.long)

    @lru_cache(maxsize=None)
    def _get_edge_index(self, num_nodes):
        '''
        Returns edge index for graph.
        - all robot nodes are connected to the previous robot node
        - all object nodes are connected to the last robot node (end-effector)
        '''
        assert len(self.eef_idx) == self.num_robots + 1
        edge_index = []
        if len(self.eef_idx) == 3: # 2 robots
            edge_index = [[self.eef_idx[0]+ 1, self.eef_idx[1] + 1]] # robot0 base link to robot1 base link
        for robot in range(self.num_robots):
            # Connectivity of all robot nodes to the previous robot node
            edge_index += [[idx, idx+1] for idx in range(self.eef_idx[robot]+ 1, self.eef_idx[robot+1])]
        # Connectivity of all other nodes to all robot nodes
        edge_index += [[node_idx, idx] for idx in range(self.eef_idx[-1] + 1, num_nodes) for node_idx in range(self.eef_idx[self.num_robots] + 1)]
            # edge_index.append(torch.tensor([node_idx, idx]) for node_idx in range(self.eef_idx[self.num_robots] + 1))
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index
    
    def _get_y_horizon(self, data, idx, horizon):
        '''
        Get y (observation) for time step t. Should contain only task-space joint positions.
        '''
        y = []
        if idx - horizon < 0:
            y.append(self.get_y_feats(data, [0]).repeat(1,horizon-idx,1)) # use fixed first observation for beginning of episode
            y.append(self.get_y_feats(data, [t for t in range(0, idx)]))
            y = torch.cat(y, dim=1)
        else: # get all observation steps with single call
            y = self.get_y_feats(data, list(range(idx - horizon, idx)))
        return y

    
    def process(self):
        idx_global = 0

        for key in tqdm(self.dataset_keys):
            episode_length = self.dataset_root[f"data/{key}/obs/object"].shape[0]
            
            for idx in range(1, episode_length - self.pred_horizon):
                
                data_raw = self.dataset_root["data"][key]["obs"]
                node_feats  = self._get_node_feats_horizon(data_raw, idx, self.pred_horizon)
                edge_index  = self._get_edge_index(node_feats.shape[0])
                edge_attrs  = self._get_edge_attrs(edge_index)
                y           = self._get_y_horizon(data_raw, idx, self.obs_horizon)
                pos         = self._get_node_pos(data_raw, idx - 1)

                data  = Data(
                    x=node_feats,
                    edge_index=edge_index,
                    edge_attr=edge_attrs,
                    y=y,
                    time=torch.tensor([idx], dtype=torch.long)/ episode_length,
                    pos=pos
                )

                torch.save(data, osp.join(self.processed_dir, f'data_{idx_global}.pt'))
                idx_global += 1

    def len(self):
        # calculate length of dataset based on self.dataset_root
        length = 0
        for key in self.dataset_keys:
            length += self.dataset_root[f"data/{key}/obs/object"].shape[0] - self.pred_horizon - self.obs_horizon - 1
        return length
    
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
    
    def get_data_stats(self, key):
        '''
        Returns min and max of data
        Used for normalizing data 
        '''
        data = []
        for idx in range(self.len()):
            data.append(torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))[key])
        data = torch.cat(data, dim=1)
        return {
            "min": torch.min(data, dim=1).values,
            "max": torch.max(data, dim=1).values
        }
    
    def normalize_data(self, data, stats_key):
        return self.normalizer[stats_key].normalize(data)
    
    def unnormalize_data(self, data, stats_key):
        return self.normalizer[stats_key].unnormalize(data)

    def get_normalizer(self):
        normalizer = LinearNormalizer()
        data_obs = []
        data_action = []
        for idx in range(self.len()):
            data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
            data_obs.append(data["y"])
            data_action.append(data["x"])
        data_obs = torch.cat(data_obs, dim=1)
        data_action = torch.cat(data_action, dim=1)
        
        normalizer.fit(
            {
                "obs": data_obs,
                "action": data_action
            }
        )
        return normalizer
    
    def to_obs_deque(self, data):
        obs_deque = collections.deque(maxlen=self.obs_horizon)
        data_t = data.clone()
        for t in range(self.obs_horizon):
            data_t.x = data.x[:,t,:]
            data_t.y = data.y[:,t,:]
            obs_deque.append(data_t.clone())
        return obs_deque
    
    def get_action(self, data):
        return data.x[:self.eef_idx[-1] + 1,:,0].T.numpy()
    