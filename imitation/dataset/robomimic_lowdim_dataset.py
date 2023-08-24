from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm

class RobomimicLowdimDataset(torch.utils.data.Dataset):
    '''
    Dataset class for SE2 task (and robomimic), with structure from robomimic.
    https://robomimic.github.io/docs/datasets/overview.html
    '''
    def __init__(self, 
                 dataset_path,
                 obs_keys,
                 pred_horizon=1,
                 obs_horizon=1,
                 action_horizon=1):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.dataset_path = dataset_path
        self.dataset_root = h5py.File(dataset_path, 'r')
        self.dataset_keys = list(self.dataset_root["data"].keys())
        try:
           self.dataset_keys.remove("mask")
        except:
              pass
        self.obs_keys = obs_keys
        self.indices = []
        self.data_at_indices = []
        self.create_sample_indices()

        self.stats = {}
        self.stats["obs"] = self.get_data_stats("obs")
        self.stats["action"] = self.get_data_stats("action")
        
    def create_sample_indices(self):
        '''
        Creates indices for sampling from dataset
        Should return all possible idx values that enables sampling of the following:
        |                idx                    |
        |-- obs_horizon --|-- action_horizon ---| * Replanning 
                          |------------ pred_horizon -------------|
        '''
        idx_global = 0
        for key in tqdm(self.dataset_keys):
            episode_length = len(self.dataset_root[f"data/{key}/obs/{self.obs_keys[0]}"])
            for idx in range(episode_length):
                if idx + self.pred_horizon > episode_length:
                    continue
                if idx - self.obs_horizon < 0:
                    continue
                self.indices.append(idx_global + idx)
                data_obs_keys = []
                for obs_key in self.obs_keys:
                    data_obs_keys.append(self.dataset_root[f"data/{key}/obs/{obs_key}"][idx-self.obs_horizon:idx+self.pred_horizon, :])
                data_obs_keys = np.concatenate(data_obs_keys, axis=-1)
                self.data_at_indices.append({
                    "obs": data_obs_keys,
                    "action": self.dataset_root[f"data/{key}/actions"][idx:idx+self.pred_horizon, :]
                })
            idx_global += episode_length
        self.indices = np.array(self.indices)


    def __len__(self):
        return len(self.indices) - 1

    def __getitem__(self, idx):
        '''
        Returns item (timestep in demo) from dataset
        '''
        return self.data_at_indices[idx]

    def get_data_stats(self, key):
        '''
        Returns min and max of data
        Used for normalizing data 
        '''
        data = []
        for d in tqdm(self.data_at_indices):
            data.append(d[key])
        data = np.concatenate(data, axis=0)
        return {
            "min": np.min(data, axis=0),
            "max": np.max(data, axis=0)
        }

        