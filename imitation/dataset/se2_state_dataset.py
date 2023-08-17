from typing import Dict, List

import h5py
import numpy as np
import torch
from tqdm import tqdm


class Se2StateDataset(torch.utils.data.Dataset):
    '''
    Dataset class for Se2 task, with structure from robomimic.
    https://robomimic.github.io/docs/datasets/overview.html
    '''
    def __init__(self, 
                 dataset_path,
                 obs_keys):
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset_root = h5py.File(dataset_path, 'r')
        self.dataset_keys = list(self.dataset_root["data"].keys())
        try:
           self.dataset_keys.remove("mask")
        except:
              pass
        self.obs_keys = obs_keys

        # inds = np.argsort([int(elem[5:]) for elem in self.dataset_keys])
        # self.demos = [self.dataset_keys[i] for i in inds]


    def __len__(self):
        # accounts for different length for each demo, with different names
        return sum([len(self.dataset_root[f"data/{key}/obs/{self.obs_keys[0]}"]) for key in self.dataset_keys])


    def __getitem__(self, idx):
        '''
        Returns item (timestep in demo) from dataset
        '''
        # print(f"trying to get item {idx}")
        idx_demo = 0
        for key in self.dataset_keys:
            if idx < len(self.dataset_root[f"data/{key}/obs/{self.obs_keys[0]}"]):
                break
            idx -= len(self.dataset_root[f"data/{key}/obs/{self.obs_keys[0]}"])
            idx_demo += 1
        idx_t = idx # timestep in demo
        key = self.dataset_keys[idx_demo]
        # print(f"key: {key}")
        data = self.dataset_root[f"data/{key}"] # demo_idx
        # turn h5py._hl.dataset.Dataset into numpy array
        obs_t = []
        # each observation modality is stored as a subgroup
        for k in data["obs"]:
            if k in self.obs_keys:
                obs_t = np.append(obs_t, data["obs/{}".format(k)][idx_t]) # numpy array

        act_t = torch.tensor(data["actions"][idx_t])
        obs_t = torch.tensor(obs_t, dtype=torch.float32)
        
        data_dict = {
            "obs": obs_t ,
            "action": act_t,
        }
        
        return data_dict

    def get_data_stats(self, data):
        '''
        Returns min and max of data
        Used for normalizing data 
        '''
        data = data.reshape(-1,data.shape[-1])
        stats = {
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0)
        }
        return stats
    

        