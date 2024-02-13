
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
                 obs_keys,
                 action_keys,
                 object_state_sizes,
                 object_state_keys,
                 pred_horizon=1,
                 obs_horizon=1,
                 action_horizon=1):
        self.obs_keys = obs_keys
        self.action_keys = action_keys
        self.pred_horizon = pred_horizon        #|
        self.obs_horizon = obs_horizon          #|} FIXED TO 1 in the graph dataset
        self.action_horizon = action_horizon    #|
        self.object_state_sizes = object_state_sizes
        self.object_state_keys = object_state_keys
        self._processed_dir = dataset_path.replace(".hdf5", "_processed")

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
        joint_positions = self._calculate_joints_positions([*data["robot0_joint_vel"][idx], *data["robot0_gripper_qpos"][idx]])
        for obs_key in self.obs_keys:
            node_feats.append(torch.tensor(data[obs_key][idx - self.obs_horizon:idx][0]))
        for dim in range(3):
            node_feats.append(joint_positions[:,dim])
            # all node features must be of same length
        NUM_OBJECTS = len(self.object_state_keys) # TODO this won't work with quaternions
        i = 0
        # create tensor of same dimension as node_feats
        obj_state_tensor = torch.zeros((NUM_OBJECTS,len(node_feats)))
        for object in range(NUM_OBJECTS):
            for obj_state in self.object_state_sizes:
                if obj_state["name"] in self.object_state_keys:
                    obj_state_tensor[object,i:i + obj_state["size"]] = torch.from_numpy(data["object"][idx - self.obs_horizon:idx][0][i:i + obj_state["size"]])
                i += obj_state["size"]

        node_feats = torch.stack(node_feats, dim=1)
        node_feats = torch.cat((node_feats, obj_state_tensor), dim=0)

        # result must be of shape (num_nodes, num_node_feats)
        return node_feats
    


    def _get_edge_index(self):
        # Adjacency matrix must be converted to COO format
        # for now, all nodes connected to next node
        edge_index = []
        for idx in range(7): # TODO connectivity of objects to robot
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
            episode_length = len(self.dataset_root[f"data/{key}/obs/{self.obs_keys[0]}"])
            edge_index = self._get_edge_index()
            for idx in range(episode_length - self.pred_horizon):
                if idx - self.obs_horizon < 0:
                    continue
                data_raw = self.dataset_root["data"][key]["obs"]
                node_feats = self._get_node_feats(data_raw, idx)
                y = self._get_y(data_raw, idx)

                data  = Data(x=node_feats,
                             edge_index=edge_index,
                             y=y)
                # if self.pre_filter is not None and not self.pre_filter(data):
                #     continue                

                torch.save(data, osp.join(self._processed_dir, f'data_{idx_global}.pt'))
                idx_global += 1

        self.__len__ = idx_global

    def len(self):
        return self.__len__
    
    def get(self, idx):
        data = torch.load(osp.join(self._processed_dir, f'data_{idx}.pt'))
        return data
    