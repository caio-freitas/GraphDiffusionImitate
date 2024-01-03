
from typing import Callable, Optional
from torch_geometric.data import Dataset, Data, InMemoryDataset
from torch_kinematics_tree.models.robots import DifferentiableFrankaPanda
import logging
import h5py
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm

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
                 action_horizon=1,
                 mode="joint-space"):
        self.mode = mode
        self.action_keys = action_keys
        self.pred_horizon = pred_horizon        #|
        self.obs_horizon = obs_horizon          #|} FIXED TO 1 in the graph dataset
        self.action_horizon = action_horizon    #|
        self.object_state_sizes = object_state_sizes
        self.object_state_keys = object_state_keys
        self.num_objects = num_objects
        self._processed_dir = dataset_path.replace(".hdf5", "_processed")

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

    def _calculate_joints_positions(self, joints):
        q = torch.tensor([joints]).to("cpu")
        q.requires_grad_(True)
        data = self.robot_fk.compute_forward_kinematics_all_links(q)
        data = data[0]
        joint_positions = []
        # TODO add joint_quaternions
        for i in range(7):
            joint_positions.append(data[i, :3, 3].detach().numpy())

        return torch.tensor(joint_positions)

    @property
    def raw_file_names(self):
        return f"robomimic_graph_dataset_{self.root}.hdf5"
    
    @property
    def processed_file_names(self):
        return f"robomimic_graph_dataset_{self.root}.pt"
    
    def _get_node_feats(self, data, idx):
        node_feats = []
        if self.mode == "end-effector":
            node_feats = torch.tensor(data["robot0_eef_pos"][idx - self.obs_horizon:idx][0])
            node_feats = node_feats.reshape(1,3)
        else:
            if self.mode == "task-space":
                node_feats = []
                for i in range(idx - self.obs_horizon, idx):
                    node_feats.append(self._calculate_joints_positions([*data["robot0_joint_pos"][i], *data["robot0_gripper_qpos"][i]]))
                node_feats = torch.stack(node_feats)
            elif self.mode == "joint-space":
                node_feats = torch.tensor(data["robot0_joint_pos"][idx - self.obs_horizon:idx][0])
                # duplicate dimensions for each joint, to match task-space - since objects are going to be represented in task-space
                # result should be of shape (7,3)
                node_feats = node_feats.repeat(3,1).transpose(0,1)
            else:
                raise NotImplementedError

                # all node features must be of same length
            node_feats = torch.stack(node_feats, dim=1)
        # add dimension for NODE_TYPE, which is 0 for robot and 1 for objects
        node_feats = torch.cat((node_feats, self.ROBOT_NODE_TYPE*torch.ones((node_feats.shape[0],1))), dim=1)
        i = 0
        
        # create tensor of same dimension as node_feats
        obj_state_tensor = torch.zeros((self.num_objects,node_feats.shape[1]- 1)) # -1 because of NODE_TYPE
        for object in range(self.num_objects):
            for obj_state in self.object_state_sizes:
                if obj_state["name"] in self.object_state_keys:
                    obj_state_tensor[object,i:i + obj_state["size"]] = torch.from_numpy(data["object"][idx - self.obs_horizon:idx][0][i:i + obj_state["size"]])
                i += obj_state["size"]

        # add dimension for NODE_TYPE, which is 0 for robot and 1 for objects
        obj_state_tensor = torch.cat((obj_state_tensor, self.OBJECT_NODE_TYPE*torch.ones((obj_state_tensor.shape[0],1))), dim=1)
        node_feats = torch.cat((node_feats, obj_state_tensor), dim=0)

        # result must be of shape (num_nodes, num_node_feats)
        return node_feats
    

    def _get_edge_attrs(self, num_nodes):
        # TODO edge attributes
        return torch.zeros((num_nodes, num_nodes))

    def _get_edge_index(self, num_nodes):
        # Adjacency matrix must be converted to COO format
        # for now, all nodes connected to next node
        edge_index = []
        for idx in range(num_nodes): # TODO connectivity of objects to robot
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
                if idx - self.obs_horizon < 0:
                    continue
                data_raw = self.dataset_root["data"][key]["obs"]
                node_feats = self._get_node_feats(data_raw, idx)
                edge_index = self._get_edge_index(node_feats.shape[0])
                edge_attrs = self._get_edge_attrs(node_feats.shape[0])
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
    