
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
from imitation.utils.generic import calculate_panda_joints_positions

log = logging.getLogger(__name__)

class RobomimicGraphDataset(InMemoryDataset):
    def __init__(self, 
                 dataset_path,
                 action_keys,
                 object_state_sizes,
                 object_state_keys,
                 pred_horizon=1,
                 obs_horizon=1,
                 node_feature_dim = 2, # joint value and node type flag
                 control_mode="JOINT_VELOCITY"):
        self.control_mode           : str = control_mode
        self.node_feature_dim       : int = node_feature_dim
        self.action_keys            : List = action_keys
        self.pred_horizon           : int = pred_horizon
        self.obs_horizon            : int = obs_horizon
        self.object_state_sizes     : Dict = object_state_sizes # can be taken from https://github.com/ARISE-Initiative/robosuite/tree/master/robosuite/environments/manipulation
        self.object_state_keys      : Dict = object_state_keys
        self.num_objects            : int = len(object_state_keys)
        self._processed_dir         : str = dataset_path.replace(".hdf5", f"_{self.control_mode}_processed_{self.obs_horizon}_{self.pred_horizon}")

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

        super().__init__(root=self._processed_dir, transform=None, pre_transform=None, pre_filter=None, log=True)
        self.stats = {}
        self.stats["y"] = self.get_data_stats("y")
        self.stats["x"] = self.get_data_stats("x")
    
    @property
    def processed_file_names(self):
        '''
        List of files in the self.processed_dir directory that need to be found in order to skip processing
        '''
        names = [f"data_{i}.pt" for i in range(self.len())]
        return names

    @lru_cache(maxsize=None)
    def _get_object_feats(self, num_objects, node_feature_dim, OBJECT_NODE_TYPE, T): # no associated joint values
        # create tensor of same dimension return super()._get_node_feats(data, t) as node_feats
        obj_state_tensor = torch.zeros((num_objects, T, node_feature_dim))
        # add dimension for NODE_TYPE, which is 0 for robot and 1 for objects
        obj_state_tensor[:,:,-1] = OBJECT_NODE_TYPE
        return obj_state_tensor

    def _get_object_pos(self, data, t):
        obj_state_tensor = torch.zeros((self.num_objects, 7)) # 3 for position, 4 for quaternion

        for object, object_state_items in enumerate(self.object_state_keys.values()):
            i = 0
            for object_state in object_state_items:
                obj_state_tensor[object,i:i + self.object_state_sizes[object_state]] = torch.from_numpy(data["object"][t][i:i + self.object_state_sizes[object_state]])
                i += self.object_state_sizes[object_state]

        return obj_state_tensor

    def _get_node_pos(self, data, t):
        node_pos = []
        node_pos.append(calculate_panda_joints_positions([*data["robot0_joint_pos"][t], *data["robot0_gripper_qpos"][t]]))
        node_pos = torch.cat(node_pos)
        obj_pos_tensor = self._get_object_pos(data, t)
        node_pos = torch.cat((node_pos, obj_pos_tensor), dim=0)
        return node_pos
    
    def _get_node_feats(self, data, t_vals):
        '''
        Calculate node features for time steps t_vals
        t_vals: list of time steps
        '''
        T = len(t_vals)
        node_feats = []
        if self.control_mode == "OSC_POSE":
            node_feats = torch.cat([torch.tensor(data["robot0_eef_pos"][t_vals]), torch.tensor(data["robot0_eef_quat"][t_vals])], dim=0)
            node_feats = node_feats.reshape(T, -1) # add dimension
        if self.control_mode == "JOINT_VELOCITY":
            node_feats = torch.cat([torch.tensor(data[f"robot0_joint_vel"][t_vals]), torch.tensor(data["robot0_gripper_qvel"][t_vals])], dim=1).T.unsqueeze(2)
        elif self.control_mode == "JOINT_POSITION":
            # [node, node_feats]
            node_feats = torch.cat([torch.tensor(data[f"robot0_joint_pos"][t_vals]), torch.tensor(data["robot0_gripper_qpos"][t_vals])], dim=1).T.unsqueeze(2)

        # add dimension for NODE_TYPE flag, which is 0 for robot and 1 for objects
        node_feats = torch.cat((node_feats, self.ROBOT_NODE_TYPE*torch.ones((node_feats.shape[0],node_feats.shape[1],1))), dim=2)
        
        obj_state_tensor = self._get_object_feats(self.num_objects, self.node_feature_dim, self.OBJECT_NODE_TYPE, T)
        
        node_feats = torch.cat((node_feats, obj_state_tensor), dim=0)

        # result must be of shape (num_nodes, num_node_feats)
        return node_feats
    
    def _get_node_feats_horizon(self, data, idx, horizon):
        '''
        Calculate node features for self.obs_horizon time steps
        '''
        node_feats = []
        episode_length = data["object"].shape[0]
        # calculate node features for timesteps idx to idx + horizon
        t_vals = list(range(idx, idx + horizon))
        node_feats = self._get_node_feats(data, t_vals)
        return node_feats
    
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


    def _get_edge_index(self, num_nodes):
        '''
        Returns edge index for graph.
        - all robot nodes are connected to the previous robot node
        - all object nodes are connected to the last robot node (end-effector)
        '''
        eef_idx = 8
        edge_index = []
        for idx in range(eef_idx):
            edge_index.append([idx, idx+1])

        # Connectivity of all other nodes to the last node of robot
        for idx in range(eef_idx + 1, num_nodes):
            edge_index.append([idx, eef_idx])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index
    
    def _get_y_horizon(self, data, idx, horizon):
        '''
        Get y (observation) for time step t. Should contain only task-space joint positions.
        '''
        y = []
        for t in range(idx, idx - horizon,-1):
            if t < 0:
                y.append(self._get_node_pos(data, 0)) # use fixed first observation for beginning of episode
            else:
                y.append(self._get_node_pos(data, t))
        return torch.stack(y, dim=1)

    
    def process(self):
        idx_global = 0

        for key in tqdm(self.dataset_keys):
            episode_length = self.dataset_root[f"data/{key}/obs/object"].shape[0]
            
            for idx in range(episode_length - self.pred_horizon):
                
                data_raw = self.dataset_root["data"][key]["obs"]
                node_feats  = self._get_node_feats_horizon(data_raw, idx, self.pred_horizon)
                edge_index  = self._get_edge_index(node_feats.shape[0])
                edge_attrs  = self._get_edge_attrs(edge_index)
                y           = self._get_y_horizon(data_raw, idx, self.obs_horizon)
                pos         = self._get_node_pos(data_raw, idx + self.pred_horizon)

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

class MultiRobotGraphDataset(RobomimicGraphDataset):
    '''
    Class to use when robomimic dataset contains multiple robots (transport task).
    '''
    def __init__(self, 
                 dataset_path,
                 action_keys,
                 object_state_sizes,
                 object_state_keys,
                 robots,
                 pred_horizon=1,
                 obs_horizon=1,
                 node_feature_dim = 8,
                 mode="joint-space"): # TODO update according to RobomimicGraphDataset
        self.num_robots = len(robots)
        self.eef_idx = [0, 7, 13]
        super().__init__(dataset_path=dataset_path,
                         action_keys=action_keys,
                         object_state_sizes=object_state_sizes,
                         object_state_keys=object_state_keys,
                         pred_horizon=pred_horizon,
                         obs_horizon=obs_horizon,
                         mode=mode,
                         node_feature_dim = node_feature_dim,
                         )
        


    def _get_node_feats(self, data, t):
        '''
        Here, robot0_eef_pos, robot1_eef_pos, ... are used as node features.
        '''
        node_feats = []
        if self.mode == "end-effector":
            for i in range(self.num_robots):
                node_feats.append(torch.cat([torch.tensor(data[f"robot{i}_eef_pos"][t - 1:t][0]), torch.tensor(data[f"robot{i}_eef_quat"][t - 1:t][0])], dim=0))
            node_feats = torch.stack(node_feats)
        elif self.mode == "task-space":
            for j in range(self.num_robots):
                node_feats.append(calculate_panda_joints_positions([*data[f"robot{j}_joint_pos"][t], *data[f"robot{j}_gripper_qpos"][t]]))
            node_feats = torch.cat(node_feats)
        elif self.mode == "joint-space":
            for i in range(self.num_robots):
                node_feats.append(torch.cat([
                    torch.tensor(data[f"robot{i}_joint_pos"][t - 1:t][0]).reshape(1,-1),
                    torch.zeros((6,7))])) # complete with zeros to match task-space dimensionality
            node_feats = torch.cat(node_feats)
        elif self.mode == "task-joint-space":
            for i in range(self.num_robots):
                node_feats.append(torch.cat([calculate_panda_joints_positions([*data[f"robot{i}_joint_pos"][t], *data[f"robot{i}_gripper_qpos"][t]]),
                                                    torch.tensor(data[f"robot{i}_joint_pos"][t - 1:t]).reshape(-1,1)], dim=1))
            node_feats = torch.cat(node_feats, dim=0)
        
        else:
            raise NotImplementedError

        # add dimension for NODE_TYPE, which is 0 for robot and 1 for objects
        node_feats = torch.cat((node_feats, self.ROBOT_NODE_TYPE*torch.ones((node_feats.shape[0],1))), dim=1)

        obj_state_tensor = self._get_object_feats(self.num_objects, self.node_feature_dim, self.OBJECT_NODE_TYPE)

        node_feats = torch.cat((node_feats, obj_state_tensor), dim=0)
        return node_feats
    
    def _get_edge_index(self, num_nodes):
        '''
        Returns edge index for graph.
        - all robot nodes are connected to the previous robot node
        - all object nodes are connected to the last robot node (end-effector)
        '''
        assert len(self.eef_idx) == self.num_robots + 1
        edge_index = []

        for id_robot in range(1, len(self.eef_idx)):
            for idx in range(self.eef_idx[id_robot-1], self.eef_idx[id_robot]):
                edge_index.append([idx, idx+1])
            # Connectivity of all other nodes to the last node of all robots
            for idx in range(self.eef_idx[self.num_robots] + 1, num_nodes):
                edge_index.append([self.eef_idx[id_robot], idx])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index
    
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
    