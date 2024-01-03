
from typing import Callable, Optional
from torch_geometric.data import Dataset, Data, InMemoryDataset
from torch_kinematics_tree.models.robots import DifferentiableFrankaPanda
import logging
import h5py
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

log = logging.getLogger(__name__)

class RobomimicGraphDataset(InMemoryDataset):
    def __init__(self, 
                 dataset_path,
                 action_keys,
                 object_state_sizes,
                 object_state_keys,
                 num_objects,
                 pred_horizon=1,
                 obs_horizon=1,
                 node_feature_dim = 8, # 3 for position, 4 for quaternion
                 mode="joint-space"):
        self.mode = mode
        self.node_feature_dim = node_feature_dim
        self.action_keys = action_keys
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.object_state_sizes = object_state_sizes # can be taken from https://github.com/ARISE-Initiative/robosuite/tree/master/robosuite/environments/manipulation
        self.object_state_keys = object_state_keys
        self.num_objects = len(object_state_keys)
        self._processed_dir = dataset_path.replace(".hdf5", f"_{self.mode}_processed")

        self.ROBOT_NODE_TYPE = 1
        self.OBJECT_NODE_TYPE = -1

        self.ROBOT_LINK_EDGE = 1
        self.OBJECT_ROBOT_EDGE = 2

        self._test_dimentionality()

        self.ROBOT_NODE_TYPE = 1
        self.OBJECT_NODE_TYPE = -1

        self.dataset_root = h5py.File(dataset_path, 'r')
        self.dataset_keys = list(self.dataset_root["data"].keys())
        try:
            self.dataset_keys.remove("mask")
        except:
            pass

        self.robot_fk = DifferentiableFrankaPanda()

        super().__init__(root=self._processed_dir, transform=None, pre_transform=None, pre_filter=None, log=True)

    def _test_dimentionality(self):
        '''
        Test if input dimensions add up to expected dimensionality
        * sum of object_state_sizes in object_state_keys must be equal to node_feature_dim
        '''
        sum_object_state_sizes = sum([self.object_state_sizes[object_state] for object_state in list(self.object_state_keys.values())[0]])
        assert self.node_feature_dim == sum_object_state_sizes + 1 # +1 for NODE_TYPE

             

        
    def _calculate_joints_positions(self, joints):
        q = torch.tensor([joints]).to("cpu")
        q.requires_grad_(True)
        data = self.robot_fk.compute_forward_kinematics_all_links(q)
        data = data[0]
        joint_positions = []
        # add joint positions
        for i in range(7):
            joint_quat = torch.tensor(R.from_matrix(data[i, :3, :3].detach().numpy()).as_quat())
            joint_positions.append(torch.cat([data[i, :3, 3], joint_quat]).reshape(1,-1))

        return torch.cat(joint_positions, dim=0)

    
    @property
    def processed_file_names(self):
        '''
        List of files in the self.processed_dir directory that need to be found in order to skip processing
        '''
        names = [f"data_{i}.pt" for i in range(self.len())]
        return names

    def _get_object_feats(self, data, t):
        # create tensor of same dimension return super()._get_node_feats(data, t) as node_feats
        if self.mode == "task-joint-space":
            obj_state_tensor = torch.zeros((self.num_objects, self.node_feature_dim + 1)) # extra dimension to match with joint position 
        else:
            obj_state_tensor = torch.zeros((self.num_objects, self.node_feature_dim))

        for object, object_state_items in enumerate(self.object_state_keys.values()):
            i = 0
            for object_state in object_state_items:
                obj_state_tensor[object,i:i + self.object_state_sizes[object_state]] = torch.from_numpy(data["object"][t - 1:t][0][i:i + self.object_state_sizes[object_state]])
                i += self.object_state_sizes[object_state]

        # add dimension for NODE_TYPE, which is 0 for robot and 1 for objects
        obj_state_tensor[:, -1] = self.OBJECT_NODE_TYPE
        return obj_state_tensor

    def _get_node_feats(self, data, t):
        node_feats = []
        if self.mode == "end-effector":
            node_feats = torch.cat([torch.tensor(data["robot0_eef_pos"][t - 1:t][0]), torch.tensor(data["robot0_eef_quat"][t - 1:t][0])], dim=0)
            node_feats = node_feats.reshape(1, -1) # add dimension
        else:
            if self.mode == "task-space":
                node_feats = []
                for i in range(t - 1, t):
                    node_feats.append(self._calculate_joints_positions([*data["robot0_joint_pos"][i], *data["robot0_gripper_qpos"][i]]))
                node_feats = torch.cat(node_feats, dim=0)
            elif self.mode == "joint-space":
                node_feats.append(torch.cat([
                    torch.tensor([*data[f"robot0_joint_pos"][t - 1:t][0], *data["robot0_gripper_qpos"][t - 1:t][0]]).reshape(1,-1),
                    torch.zeros((6,9))])) # complete with zeros to match task-space dimensionality
                node_feats = torch.cat(node_feats).T
            elif self.mode == "task-joint-space":
                node_feats = []
                node_feats.append(torch.cat([self._calculate_joints_positions([*data["robot0_joint_pos"][t], *data["robot0_gripper_qpos"][t]]),
                                                torch.tensor(data["robot0_joint_pos"][t - 1:t]).reshape(-1,1)], dim=1))
                node_feats = torch.cat(node_feats, dim=0)
        # add dimension for NODE_TYPE, which is 0 for robot and 1 for objects
        node_feats = torch.cat((node_feats, self.ROBOT_NODE_TYPE*torch.ones((node_feats.shape[0],1))), dim=1)
        
        obj_state_tensor = self._get_object_feats(data, t)
        
        node_feats = torch.cat((node_feats, obj_state_tensor), dim=0)

        # result must be of shape (num_nodes, num_node_feats)
        return node_feats
    
    def _get_node_feats_horizon(self, data, idx, horizon):
        '''
        Calculate node features for self.obs_horizon time steps
        '''
        node_feats = []
        for t in range(idx - horizon + 1, idx + 1):
            node_feats.append(self._get_node_feats(data, t))
        return torch.stack(node_feats, dim=1)

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

    def _get_edge_attrs(self, num_nodes):
        # TODO edge attributes
        return torch.zeros((num_nodes, num_nodes))

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
    
    def _get_y(self, data, idx):
        actions = []
        for action_key in self.action_keys:
            actions.append(torch.tensor(data[action_key][idx:idx + self.pred_horizon]))
        
        return torch.stack(actions, dim=-1)

    def process(self):
        idx_global = 0

        for key in tqdm(self.dataset_keys):
            episode_length = self.dataset_root[f"data/{key}/obs/object"].shape[0]
            
            for idx in range(episode_length - self.pred_horizon):
                if idx - self.obs_horizon <= 0 or idx + self.pred_horizon > episode_length:
                    continue
                data_raw = self.dataset_root["data"][key]["obs"]
                node_feats  = self._get_node_feats_horizon(data_raw, idx + self.pred_horizon, self.pred_horizon)
                edge_index  = self._get_edge_index(node_feats.shape[0])
                edge_attrs  = self._get_edge_attrs(edge_index)
                y           = self._get_node_feats_horizon(data_raw, idx, self.obs_horizon)

                data  = Data(x=node_feats,
                             edge_index=edge_index,
                             edge_attr=edge_attrs,
                             y=y)             

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
                 mode="joint-space"):
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
                node_feats.append(self._calculate_joints_positions([*data[f"robot{j}_joint_pos"][t], *data[f"robot{j}_gripper_qpos"][t]]))
            node_feats = torch.cat(node_feats)
        elif self.mode == "joint-space":
            for i in range(self.num_robots):
                node_feats.append(torch.cat([
                    torch.tensor(data[f"robot{i}_joint_pos"][t - 1:t][0]).reshape(1,-1),
                    torch.zeros((6,7))])) # complete with zeros to match task-space dimensionality
            node_feats = torch.cat(node_feats)
        elif self.mode == "task-joint-space":
            for i in range(self.num_robots):
                node_feats.append(torch.cat([self._calculate_joints_positions([*data[f"robot{i}_joint_pos"][t], *data[f"robot{i}_gripper_qpos"][t]]),
                                                    torch.tensor(data[f"robot{i}_joint_pos"][t - 1:t]).reshape(-1,1)], dim=1))
            node_feats = torch.cat(node_feats, dim=0)
        
        else:
            raise NotImplementedError

        # add dimension for NODE_TYPE, which is 0 for robot and 1 for objects
        node_feats = torch.cat((node_feats, self.ROBOT_NODE_TYPE*torch.ones((node_feats.shape[0],1))), dim=1)

        obj_state_tensor = self._get_object_feats(data, t)

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
        