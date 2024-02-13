import torch
from torch import nn

from imitation.policy.base_policy import BasePolicy
from imitation.model.mlp import MLPNet

import logging
import wandb
import os
from tqdm.auto import tqdm

from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
from diffusion_policy.common.robomimic_config_util import get_robomimic_config

log = logging.getLogger(__name__)

class RobomimicLowdimPolicy(BasePolicy):
    def __init__(self, 
            action_dim, 
            obs_dim,
            algo_name='bc',
            obs_type='low_dim',
            task_name='square',
            dataset_type='ph',
            dataset=None,
            ckpt_path=None,
            lr=1e-4
        ):
        super().__init__()
        # key for robomimic obs input
        # previously this is 'object', 'robot0_eef_pos' etc
        self.dataset = dataset
        self.lr = lr
        obs_key = 'obs'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device {self.device}")
        config = get_robomimic_config(
            algo_name=algo_name,
            hdf5_type=obs_type,
            task_name=task_name,
            dataset_type=dataset_type)
        with config.unlocked():
            config.observation.modalities.obs.low_dim = [obs_key]
        
        ObsUtils.initialize_obs_utils_with_config(config)
        model: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes={obs_key: [obs_dim]},
                ac_dim=action_dim,
                device=self.device,
            )
        self.model = model
        self.nets = model.nets
        self.obs_key = obs_key
        self.config = config

        # load model from ckpt
        if ckpt_path is not None:
            self.load_nets(ckpt_path)
        # create dataloader
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=32,
            num_workers=1,
            shuffle=True,
            # accelerate cpu-gpu transfer
            pin_memory=True,
            # don't kill worker process afte each epoch
            persistent_workers=True,
        )

        self.global_epoch = 0

    def to(self,*args,**kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.model.device = device
        super().to(*args,**kwargs)
    
    def load_nets(self, ckpt_path):
        log.info(f"Loading model from {ckpt_path}")
        try:
            self.model.deserialize(torch.load(ckpt_path))
        except Exception as e:
            log.error(f"Could not load model from {ckpt_path}. Error: {e}")

    # =========== inference =============
    def get_action(self, obs):
        if self.model.nets.training:        
            self.model.set_eval()
        obs = torch.tensor([obs], dtype=torch.float32).to(self.device)
        robomimic_obs_dict = {self.obs_key: obs[:,0,:]}
        action = self.model.get_action(robomimic_obs_dict)
        # (B, Da)
        return action.cpu().numpy()
    
    def reset(self):
        self.model.reset()
        
    # =========== training ==============
    
    def train(self, dataset, num_epochs, model_path, seed=0):
        with tqdm(range(num_epochs)) as pbar:
            for epoch in pbar:
                for batch in self.dataloader:
                    robomimic_batch = {
                        'obs': {self.obs_key: batch['obs']},
                        'actions': batch['action']
                    }
                    input_batch = self.model.process_batch_for_training(
                        robomimic_batch)
                    info = self.model.train_on_batch(
                        batch=input_batch, epoch=epoch, validate=False)
                wandb.log({"epoch": self.global_epoch,
                        "loss": info['losses']['action_loss'],
                        "log_probs": info['losses']['log_probs']})
                # save model from dict in self.model.serialize()
                torch.save(self.model.serialize(), model_path)
                self.global_epoch += 1
                
        torch.save(self.model.serialize(), model_path + f'{num_epochs}_ep.pt')
        
        return info
