from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import os
import logging
from diffusion_policy.model.common.normalizer import LinearNormalizer


from diffusion_policy.model.common.rotation_transformer import RotationTransformer


log = logging.getLogger(__name__)

class RobomimicLowdimDataset(torch.utils.data.Dataset):
    '''
    Dataset class for SE2 task (and robomimic), with structure from robomimic.
    https://robomimic.github.io/docs/datasets/overview.html
    '''
    def __init__(self, 
                 dataset_path,
                 obs_keys,
                 action_keys,
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
        self.rotation_transformer = RotationTransformer(
            from_rep="quaternion",
            to_rep="rotation_6d"
        )
        try:
           self.dataset_keys.remove("mask")
        except:
              pass
        self.obs_keys = obs_keys
        self.action_keys = action_keys
        self.indices = []
        self.data_at_indices = []
        # if indices file exists, load it
        index_file = dataset_path.replace(".hdf5", f"_indices_{obs_horizon}_{action_horizon}_{pred_horizon}.npy")
        data_at_indices_file = dataset_path.replace(".hdf5", f"_data_at_indices_{obs_horizon}_{action_horizon}_{pred_horizon}.npy")
        if os.path.exists(index_file):
            self.indices = np.load(index_file)
            self.data_at_indices = np.load(data_at_indices_file, allow_pickle=True)
        else:
            self.create_sample_indices()
            # keeps sample indices in memory, so that it doesn't need to be reloaded
            np.save(index_file, self.indices)
            np.save(data_at_indices_file, self.data_at_indices)

        self.stats = {}
        self.stats["obs"] = self.get_data_stats("obs")
        self.stats["action"] = self.get_data_stats("action")
        
    def get_normalizer(self):
        normalizer = LinearNormalizer()
        data_obs = []
        for d in tqdm(self.data_at_indices):
            data_obs.append(d["obs"])
        data_obs = np.concatenate(data_obs, axis=0)
        data_action = []
        for d in tqdm(self.data_at_indices):
            data_action.append(d["action"])
        data_action = np.concatenate(data_action, axis=0)
        self.stats["obs"] = data_obs
        self.stats["action"] = data_action
        normalizer.fit(self.stats)
        return normalizer


    def create_sample_indices(self):
        '''
        Creates indices for sampling from dataset
        Should return all possible idx values that enables sampling of the following:
        |                idx                    |
        |-- obs_horizon --|-- action_horizon ---| * Replanning 
                          |------------ pred_horizon -------------|
        '''
        idx_global = 0
        n_latency_steps = 0
        for key in tqdm(self.dataset_keys):
            episode_length = len(self.dataset_root[f"data/{key}/obs/{self.obs_keys[0]}"])
            for idx in range(episode_length - self.pred_horizon + n_latency_steps):
                if idx - self.obs_horizon < 0:
                    continue
                self.indices.append(idx_global + idx)
                data_obs_keys = []
                for obs_key in self.obs_keys:
                    obs = self.dataset_root[f"data/{key}/obs/{obs_key}"][idx - self.obs_horizon:idx, :]
                    if "quat" in obs_key:
                        obs = self.rotation_transformer.forward(obs)
                    data_obs_keys.append(obs)
                data_action_keys = []
                for action_key in self.action_keys:
                    action = self.dataset_root[f"data/{key}/obs/{action_key}"][idx + n_latency_steps:idx + n_latency_steps + self.pred_horizon, :]
                    if "quat" in action_key:
                        action = self.rotation_transformer.forward(action)
                    data_action_keys.append(action)
                data_obs_keys = np.concatenate(data_obs_keys, axis=-1)
                data_action_keys = np.concatenate(data_action_keys, axis=-1)
                self.data_at_indices.append({
                    "obs": data_obs_keys,
                    "action": data_action_keys
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

        