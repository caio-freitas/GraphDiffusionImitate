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
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from diffusion_policy.model.common.normalizer import LinearNormalizer

log = logging.getLogger(__name__)

class RobomimicPretrainedWrapper:
    def __init__(self,
            action_dim,
            obs_dim,
            algo_name='bc_rnn',
            obs_type='low_dim',
            task_name='square',
            dataset_type='ph',
            dataset=None,
            ckpt_path=None,
            lr=1e-4
        ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load model from ckpt
        if ckpt_path is not None:
            self.load_nets(ckpt_path)

    def load_nets(self, ckpt_path):
        log.info(f"Loading model from {ckpt_path}")
        try:
            policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=self.device, verbose=True)
        except Exception as e:
            log.error(f"Could not load model from {ckpt_path}. Error: {e}")
            raise e
        policy.start_episode()
        self.policy = policy

    def get_action(self, obs):
        obs = torch.tensor(obs).to(self.device).squeeze(0)
        obs_dict = {
            "robot0_eef_pos": obs[:3],
            "robot0_eef_quat": obs[3:7],
            "robot0_gripper_qpos": obs[7:9],
            "object": obs[9:]
        }
        return self.policy(ob=obs_dict)

class RobomimicLowdimPolicy(BasePolicy):
    def __init__(self, 
            action_dim, 
            obs_dim,
            algo_name='bc_rnn',
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
        assert self.dataset.pred_horizon == self.dataset.obs_horizon == self.dataset.action_horizon, \
            "Robomimic only supports pred_horizon == obs_horizon"
        self.lr = lr
        obs_key = 'obs'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device {self.device}")
        config = get_robomimic_config(
            algo_name=algo_name,
            hdf5_type=obs_type,
            task_name=task_name,
            dataset_type=dataset_type,)
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
        self.nets = self.model.nets
        self.normalizer = self.dataset.get_normalizer()
        self.obs_key = obs_key
        self.config = config

        # load model from ckpt
        if ckpt_path is not None:
            self.load_nets(ckpt_path)

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
        obs = self.normalizer['obs'].normalize(torch.tensor(obs)).to(self.device)
        if self.model.nets.training:        
            self.model.set_eval()
        robomimic_obs_dict = {self.obs_key: obs[:,:]}
        naction = self.model.get_action(robomimic_obs_dict)
        action = self.normalizer['action'].unnormalize(naction)
        # (B, Da)
        return action.cpu().numpy()
    
    def reset(self):
        self.model.reset()
        
    # =========== training ==============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def validate(self, dataset, model_path):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            num_workers=1,
            shuffle=False,
            # accelerate cpu-gpu transfer
            pin_memory=True,
            # don't kill worker process afte each epoch
            persistent_workers=True,
        )
        val_loss = 0
        for batch in dataloader:
            nbatch = self.normalizer.normalize(batch)
            assert len(nbatch["action"].shape) == 3
            robomimic_batch = {
                'obs': {self.obs_key: nbatch['obs']},
                'actions': nbatch['action']
            }
            input_batch = self.model.process_batch_for_training(
                robomimic_batch)
            info = self.model.train_on_batch(
                batch=input_batch, epoch=self.global_epoch, validate=True)
            wandb.log({"val_loss": info['losses']['action_loss']})
            val_loss += info['losses']['action_loss']
        val_loss /= len(dataloader)
        return val_loss

    def train(self, dataset, num_epochs, model_path, seed=0):
        # create dataloader
        self.model.nets.train()
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            num_workers=1,
            shuffle=False,
            # accelerate cpu-gpu transfer
            pin_memory=True,
            # don't kill worker process afte each epoch
            persistent_workers=True,
        )
        with tqdm(range(num_epochs)) as pbar:
            for epoch in pbar:
                for batch in dataloader:
                    nbatch = self.normalizer.normalize(batch)
                    assert len(nbatch["action"].shape) == 3
                    robomimic_batch = {
                        'obs': {self.obs_key: nbatch['obs']},
                        'actions': nbatch['action']
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
    
    def get_optimizer(self):
        return self.model.optimizers['policy']

