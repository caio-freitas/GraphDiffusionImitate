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
                 pred_horizon=1):
        super().__init__()
        self.pred_horizon = pred_horizon
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
        # and considers pred_horizon
        return sum(
            [len(self.dataset_root[f"data/{key}/obs/{self.obs_keys[0]}"]) for key in self.dataset_keys]
        ) - self.pred_horizon*len(self.dataset_keys)

    def __getitem__(self, idx):
        '''
        Returns item (timestep in demo) from dataset
        '''
        idx_demo = 0
        # find which demo idx is in, using self.pred_horizon steps in the future
        while idx + self.pred_horizon > len(self.dataset_root[f"data/{self.dataset_keys[idx_demo]}/obs/{self.obs_keys[0]}"]):
            if idx <= 0 or idx_demo == len(self.dataset_keys):
                print("idx out of bounds")
                print(f"idx: {idx}, idx_demo: {idx_demo}")
                
                return None
            idx -= min(len(self.dataset_root[f"data/{self.dataset_keys[idx_demo]}/obs/{self.obs_keys[0]}"]), idx)
            # the data will be repeated for all idx in the range len(demos) - self.pred_horizon
            idx_demo += 1

            
        assert idx_demo < len(self.dataset_keys), f"idx_demo: {idx_demo}, len(self.dataset_keys): {len(self.dataset_keys)}"
        assert idx <= len(self.dataset_root[f"data/{self.dataset_keys[idx_demo]}/obs/{self.obs_keys[0]}"]), f"idx: {idx}, len(self.dataset_root[f'data/{self.dataset_keys[idx_demo]}/obs/{self.obs_keys[0]}']): {len(self.dataset_root[f'data/{self.dataset_keys[idx_demo]}/obs/{self.obs_keys[0]}'])}"
        
        idx_t = idx # initial timestep in demo
        key = self.dataset_keys[idx_demo]
        demo = self.dataset_root[f"data/{key}"] # demo_idx
        obs_len = len(demo["obs/{}".format(self.obs_keys[0])][0])
        key_len = len(self.obs_keys)   
        obs_t = None # (pred_horizon, key_len*obs_len)
        
        # for each observation modality, store pred_horizon steps in the future
        # to turn h5py._hl.dataset.Dataset into numpy array
        for i, obs_key in enumerate(self.obs_keys):
            if i == 0:
                obs_t = torch.tensor(demo[f"obs/{obs_key}"][idx_t:idx_t+self.pred_horizon, :], dtype=torch.float32)
                continue
            obs_t = torch.cat(
                (obs_t,
                torch.tensor(demo[f"obs/{obs_key}"][idx_t:idx_t+self.pred_horizon, :], dtype=torch.float32)),
                dim=-1                
                )

        obs_t = obs_t.flatten()

        act_t = torch.tensor(demo["actions"][idx_t:idx_t+self.pred_horizon], dtype=torch.float32)
        act_t = act_t.flatten()

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
    

        