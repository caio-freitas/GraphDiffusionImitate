
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
                 pred_horizon=1,
                 obs_horizon=1,
                 action_horizon=1):
        self.obs_keys = obs_keys
        self.action_keys = action_keys
        self.pred_horizon = pred_horizon        #|
        self.obs_horizon = obs_horizon          #|} FIXED TO 1 in the graph dataset
        self.action_horizon = action_horizon    #|
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
        print(q)
        q.requires_grad_(True)
        data = self.robot_fk.compute_forward_kinematics_all_links(q)
        data.shape
        joint_positions = []
        for i in range(7):
            joint_positions.append(data[:, :3, 3].detach().numpy())
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
            node_feats.append(torch.tensor(data[obs_key][idx - self.obs_horizon:idx]))
        for node in range(len(node_feats.shape[0])):
            node_feats.append(joint_positions[node])
            # TODO all node features must be of same length

        # result must be of shape (num_nodes, num_node_feats)
        return torch.stack(node_feats, dim=1)[0]


    def _get_edge_index(self):
        # Adjacency matrix must be converted to COO format
        # for now, all nodes connected to next node
        edge_index = []
        for idx in range(6):
            edge_index.append([idx, idx+1])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index
    
    def _get_y(self, data, idx):
        actions = []
        for action_key in self.action_keys:
            actions.append(torch.tensor(data[action_key][idx:idx + self.pred_horizon]))
        
        return torch.stack(actions, dim=1)

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

                data  = Data(x=np.transpose(node_feats),
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
    