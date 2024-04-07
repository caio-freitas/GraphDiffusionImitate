
from typing import Callable, Optional
from torch_geometric.data import Dataset, Data, InMemoryDataset, HeteroData
import logging
import h5py
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict
from functools import lru_cache
from scipy.spatial.transform import Rotation as R

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
        self.obs_mode               : str = "OSC_POSE"
        self.robots                 : List = robots
        self.num_robots             : int = len(self.robots)
        self.node_feature_dim       : int = 8 if control_mode == "OSC_POSE" else node_feature_dim
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
        self.stats = {}
        self.stats["y"] = self.get_data_stats("y")
        self.stats["x"] = self.get_data_stats("x")

        self.constant_stats = {
            "y": torch.tensor([False, False, False, True, True, True, True, True, True]), # mask rotations for robot and object nodes
            "x": torch.tensor([False, False, False, True, True, True, True, True, True, True]) # node type flag is constant
        }
        

    @property
    def processed_file_names(self):
        '''
        List of files in the self.processed_dir directory that need to be found in order to skip processing
        '''
        names = [f"data_{i}.pt" for i in range(self.len())]
        return names

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

    def _get_node_pos(self, data, t, modality="action"):
        node_pos = []
        for i in range(self.num_robots):
            if self.control_mode == "OSC_POSE":
                node_pos_robot = torch.cat([torch.tensor(data["robot0_eef_pos"][t]), torch.tensor(data["robot0_eef_quat"][t])], dim=0).unsqueeze(0)
            else:
                node_pos_robot = calculate_panda_joints_positions([*data["robot0_joint_pos"][t], *data["robot0_gripper_qpos"][t]])
                node_pos_robot[:,:3] += torch.tensor(self.BASE_LINK_SHIFT)
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
        if modality == "observation":
            obj_pos_tensor = self._get_object_pos(data, t)
            node_pos = torch.cat((node_pos, obj_pos_tensor), dim=0)
        return node_pos
    
    def _get_node_actions(self, data, t_vals, modality="action"):
        '''
        Calculate node features for time steps t_vals
        t_vals: list of time steps
        '''
        T = len(t_vals)
        node_feats = []
        if self.control_mode == "OSC_POSE":
            for i in range(self.num_robots):
                node_feats.append(torch.cat([torch.tensor(data["actions"][t_vals])], dim=1).unsqueeze(0))
        elif self.control_mode == "JOINT_POSITION":
            for i in range(self.num_robots):
                node_feats.append(torch.cat([
                                torch.tensor(data[f"robot{i}_joint_pos"][t_vals]),
                                torch.tensor(data[f"robot{i}_gripper_qpos"][t_vals])], dim=1).T.unsqueeze(2))
        elif self.control_mode == "JOINT_VELOCITY":
            for i in range(self.num_robots):
                node_feats.append(torch.cat([
                                torch.tensor(data[f"robot{i}_joint_vel"][t_vals]),
                                torch.tensor(data[f"robot{i}_gripper_qvel"][t_vals])], dim=1).T.unsqueeze(2))
        node_feats = torch.cat(node_feats, dim=0) # [num_robots*num_joints, T, 1]

        return node_feats
    
    def _get_node_actions_horizon(self, data, idx, horizon):
        '''
        Calculate node features for self.obs_horizon time steps
        '''
        node_feats = []
        # calculate node features for timesteps idx to idx + horizon
        t_vals = list(range(idx, idx + horizon))
        node_feats = self._get_node_actions(data, t_vals)
        return node_feats
    
    # @lru_cache(maxsize=None)
    def _get_edge_attrs(self, edge_index, modality="action"):
        '''
        Attribute edge types to edges
        - self.ROBOT_LINK_EDGE for edges between robot nodes
        - self.OBJECT_ROBOT_EDGE for edges between robot and object nodes
        '''
        edge_attrs = []
        
        if modality == "action":
            return torch.ones(edge_index.shape[1], dtype=torch.long) * self.ROBOT_LINK_EDGE

        num_nodes = torch.max(edge_index).item()
        for edge in edge_index.T:
            # num nodes - self.num_objects is the index of the last robot node
            if edge[0] <= num_nodes - self.num_objects and edge[1] <= num_nodes - self.num_objects:
                edge_attrs.append(self.ROBOT_LINK_EDGE)
            # there are no object-to-object edges
            else:
                edge_attrs.append(self.OBJECT_ROBOT_EDGE)
        return torch.tensor(edge_attrs, dtype=torch.long)

    # @lru_cache(maxsize=None)
    def _get_edge_index(self, num_nodes, modality="action"):
        '''
        Returns edge index for graph.
        - all robot nodes are connected to the previous robot node
        - all object nodes are connected to the last robot node (end-effector)
        '''
        edge_index = []
        graph_type = self.control_mode if modality == "action" else self.obs_mode
        eef_idx = self.eef_idx
        if graph_type == "OSC_POSE": # 1 node per robot
            eef_idx = list(range(-1, self.num_robots))
            edge_index += [[0, 0]]
            edge_index += [[eef_idx[robot_node], eef_idx[robot_node+1]] for robot_node in range(1, len(eef_idx)-1)]
        else: # JOINT_POSITION or JOINT_VELOCITY
            assert len(self.eef_idx) == self.num_robots + 1

            if len(self.eef_idx) == 3: # 2 robots
                edge_index = [[self.eef_idx[0]+ 1, self.eef_idx[1] + 1]] # robot0 base link to robot1 base link
            for robot in range(self.num_robots):
                # Connectivity of all robot nodes to the previous robot node
                edge_index += [[idx, idx+1] for idx in range(self.eef_idx[robot]+ 1, self.eef_idx[robot+1])]
            
        if modality == "observation":
            # Connectivity of all objects to all robot nodes
            edge_index += [[node_idx, idx] for idx in range(eef_idx[-1] + 1, num_nodes) for node_idx in range(eef_idx[self.num_robots] + 1)]

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index
    
    def _get_obs_horizon(self, data, idx, horizon, modality="observation"):
        '''
        Get observation node features for time step t. Should contain only task-space joint positions.
        '''
        obs = []
        for t in range(idx, idx - horizon,-1):
            if t < 0:
                obs.append(self._get_node_pos(data, 0, modality="observation")) # use fixed first observation for beginning of episode
            else:
                obs.append(self._get_node_pos(data, t, modality="observation"))
        return torch.stack(obs, dim=1)

    
    def process(self):
        idx_global = 0

        for key in tqdm(self.dataset_keys):
            episode_length = self.dataset_root[f"data/{key}/obs/object"].shape[0]
            
            for idx in range(episode_length - self.pred_horizon):
                
                data_raw = self.dataset_root["data"][key]
                action_feats  = self._get_node_actions_horizon(data_raw, idx, self.pred_horizon)
                action_edge_index  = self._get_edge_index(action_feats.shape[0], modality="action")
                action_edge_attrs  = self._get_edge_attrs(action_edge_index, modality="action")
                action_pos = self._get_node_pos(data_raw["obs"], idx + self.pred_horizon, modality="action")
                
                obs_feats = self._get_obs_horizon(data_raw["obs"], idx, self.obs_horizon)
                obs_pos = self._get_node_pos(data_raw["obs"], idx, modality="observation")
                obs_edge_index = self._get_edge_index(obs_feats.shape[0], modality="observation")
                obs_edge_attrs = self._get_edge_attrs(obs_edge_index, modality="observation")

                action_graph  = Data(
                    x=action_feats,
                    edge_index=action_edge_index,
                    edge_attr=action_edge_attrs,
                    y=action_pos,
                    time=torch.tensor([idx], dtype=torch.long)/ episode_length,
                    pos=action_pos
                )
                obs_graph = Data(
                    x=obs_feats,
                    edge_index=obs_edge_index,
                    edge_attr=obs_edge_attrs,
                    time=torch.tensor([idx], dtype=torch.long)/ episode_length,
                    pos=obs_pos
                )

                data = HeteroData()
                data["action"] = action_graph
                data["observation"] = obs_graph

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
    
    def normalize_data(self, data, stats_key, batch_size=1):
        # avoid division by zero by skipping normalization
        with torch.no_grad():
            # duplicate stats for each batch
            data = data.clone().to(dtype=torch.float64)
            stats = self.stats[stats_key].copy()
            stats["min"] = stats["min"].repeat(batch_size, 1)
            stats["max"] = stats["max"].repeat(batch_size, 1)
            to_normalize = ~self.constant_stats[stats_key]
            constant_stats = stats["max"] == stats["min"]
            stats["min"][constant_stats] = -1
            stats["max"][constant_stats] = 1
            for t in range(data.shape[1]):
                data[:,t,to_normalize] = (data[:,t,to_normalize] - stats['min'][:,to_normalize]) / (stats['max'][:,to_normalize] - stats['min'][:,to_normalize])
                data[:,t,to_normalize] = data[:,t,to_normalize] * 2 - 1
        return data

    def unnormalize_data(self, data, stats_key, batch_size=1):
        # avoid division by zero by skipping normalization
        with torch.no_grad():
            stats = self.stats[stats_key].copy()
            # duplicate stats for each batch
            stats["min"] = stats["min"].repeat(batch_size, 1)
            stats["max"] = stats["max"].repeat(batch_size, 1)
            data = data.clone().to(dtype=torch.float64)
            to_normalize = ~self.constant_stats[stats_key]
            constant_stats = stats["max"] == stats["min"]
            stats["min"][constant_stats] = -1
            stats["max"][constant_stats] = 1
            for t in range(data.shape[1]):
                data[:,t,to_normalize] = (data[:,t,to_normalize] + 1) / 2
                data[:,t,to_normalize] = data[:,t,to_normalize] * (stats['max'][:,to_normalize] - stats['min'][:,to_normalize]) + stats['min'][:,to_normalize]
        return data
