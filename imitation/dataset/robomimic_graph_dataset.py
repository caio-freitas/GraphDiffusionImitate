
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
                 pred_horizon=1,
                 obs_horizon=1,
                 action_horizon=1,
                 node_feature_dim = 8, # 3 for position, 4 for quaternion
                 mode="joint-space"):
        self.mode = mode
        self.node_feature_dim = node_feature_dim
        self.action_keys = action_keys
        self.pred_horizon = pred_horizon        #|
        self.obs_horizon = obs_horizon          #| TODO use horizons
        self.action_horizon = action_horizon    #|
        self.object_state_sizes = object_state_sizes # can be taken from https://github.com/ARISE-Initiative/robosuite/tree/master/robosuite/environments/manipulation
        self.object_state_keys = object_state_keys
        self.num_objects = len(object_state_keys)
        self._processed_dir = dataset_path.replace(".hdf5", "_processed")

        self.ROBOT_NODE_TYPE = 1
        self.OBJECT_NODE_TYPE = -1

        self._test_dimentionality()

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
    def raw_file_names(self):
        return f"robomimic_graph_dataset_{self.root}.hdf5"
    
    @property
    def processed_file_names(self):
        return f"robomimic_graph_dataset_{self.root}.pt"
    
    def _get_object_feats(self, data, t):
        # create tensor of same dimension return super()._get_node_feats(data, t) as node_feats
        obj_state_tensor = torch.zeros((self.num_objects, self.node_feature_dim)) # -1 because of NODE_TYPE
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
                    torch.tensor(data[f"robot0_joint_pos"][t - 1:t][0]).reshape(1,-1),
                    torch.zeros((6,7))])) # complete with zeros to match task-space dimensionality
                node_feats = torch.cat(node_feats)
            else:
                raise NotImplementedError

        # add dimension for NODE_TYPE, which is 0 for robot and 1 for objects
        node_feats = torch.cat((node_feats, self.ROBOT_NODE_TYPE*torch.ones((node_feats.shape[0],1))), dim=1)
        
        obj_state_tensor = self._get_object_feats(data, t)
        
        node_feats = torch.cat((node_feats, obj_state_tensor), dim=0)

        # result must be of shape (num_nodes, num_node_feats)
        return node_feats
    

    def _get_edge_attrs(self, edge_index):
        return torch.ones((edge_index.shape[1])) # TODO edge attributes

    def _get_edge_index(self, num_nodes):
        # Adjacency matrix must be converted to COO format
        # for now, all nodes connected to next node
        edge_index = []
        for idx in range(num_nodes - 1): # TODO connectivity of objects to robot
            edge_index.append([idx, idx+1])

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
                if idx - 1 < 0 or idx + self.pred_horizon > episode_length:
                    continue
                data_raw = self.dataset_root["data"][key]["obs"]
                node_feats = self._get_node_feats(data_raw, idx)
                edge_index = self._get_edge_index(node_feats.shape[0])
                edge_attrs = self._get_edge_attrs(edge_index)
                y = self._get_y(data_raw, idx)

                data  = Data(x=node_feats,
                             edge_index=edge_index,
                             edge_attr=edge_attrs,
                             y=y)             

                torch.save(data, osp.join(self._processed_dir, f'data_{idx_global}.pt'))
                idx_global += 1

        self.__len__ = idx_global

    def len(self):
        return self.__len__
    
    def get(self, idx):
        data = torch.load(osp.join(self._processed_dir, f'data_{idx}.pt'))
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
                 action_horizon=1,
                 node_feature_dim = 8,
                 mode="joint-space"):
        self.num_robots = len(robots)
        super().__init__(dataset_path=dataset_path,
                         action_keys=action_keys,
                         object_state_sizes=object_state_sizes,
                         object_state_keys=object_state_keys,
                         pred_horizon=pred_horizon,
                         obs_horizon=obs_horizon,
                         action_horizon=action_horizon,
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
        else:
            raise NotImplementedError

        # add dimension for NODE_TYPE, which is 0 for robot and 1 for objects
        node_feats = torch.cat((node_feats, self.ROBOT_NODE_TYPE*torch.ones((node_feats.shape[0],1))), dim=1)

        obj_state_tensor = self._get_object_feats(data, t)

        node_feats = torch.cat((node_feats, obj_state_tensor), dim=0)
        return node_feats