import torch
import os
import gdown
import numpy as np
import logging

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


from imitation.policy.base_policy import BasePolicy
from imitation.dataset.pusht_state_dataset import PushTStateDataset
from imitation.model.diffusion_policy.conditional_unet1d import ConditionalUnet1D


log = logging.getLogger(__name__)

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data



class DiffusionUnet1DPolicy(BasePolicy):
    def __init__(self, 
                    env,
                    obs_dim = 5,
                    action_dim = 2,
                    pred_horizon = 16,
                    obs_horizon = 2,
                    action_horizon = 2,
                    num_diffusion_iters = 1000):
        super().__init__(env)
        self.env = env
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.num_diffusion_iters = num_diffusion_iters

        self.device = torch.device("cpu") # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )
        

        self._load_nets()
        self._init_stats()

    def _load_nets(self):
        # create network object
        noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim*self.obs_horizon,
            kernel_size=5
        )
        # load from checkpoint
        ckpt_path = "pusht_state_100ep.ckpt"
        if not os.path.isfile(ckpt_path):
            id = "1mHDr_DEZSdiGo9yecL50BBQYzR8Fjhl_&confirm=t"
            gdown.download(id=id, output=ckpt_path, quiet=False)

        state_dict = torch.load(ckpt_path, map_location='cuda')
        self.ema_noise_pred_net = noise_pred_net
        self.ema_noise_pred_net.load_state_dict(state_dict)
        print('Pretrained weights loaded.')

    def _init_stats(self):
        logging.log(logging.INFO, "Initializing data statistics from given dataset")
        # download demonstration data from Google Drive
        dataset_path = "pusht_cchi_v7_replay.zarr.zip"
        if not os.path.isfile(dataset_path):
            id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
            gdown.download(id=id, output=dataset_path, quiet=False)

        # create dataset from file
        dataset = PushTStateDataset(
            dataset_path=dataset_path,
            pred_horizon=self.pred_horizon,
            obs_horizon=self.obs_horizon,
            action_horizon=self.action_horizon
        )
        # save training data statistics (min, max) for each dim
        self.stats = dataset.stats

        # create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=256,
            num_workers=1,
            shuffle=True,
            # accelerate cpu-gpu transfer
            pin_memory=True,
            # don't kill worker process afte each epoch
            persistent_workers=True
        )

        # visualize data in batch
        batch = next(iter(dataloader))
        logging.log(logging.INFO, f"batch['obs'].shape:{batch['obs'].shape}")
        logging.log(logging.INFO, f"batch['action'].shape: {batch['action'].shape}")

    def get_action(self, obs_seq):
        B = 1 # action shape is (B, Ta, Da), observations (B, To, Do)
        nobs = normalize_data(obs_seq, stats=self.stats['obs'])
        # device transfer
        nobs = torch.from_numpy(nobs).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)
            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, self.pred_horizon, self.action_dim), device=self.device)
            naction = noisy_action

            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.ema_noise_pred_net(
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

        # unnormalize action
        naction = naction.detach().to('cpu').numpy()
        # (B, pred_horizon, action_dim)
        naction = naction[0]
        action_pred = unnormalize_data(naction, stats=self.stats['action'])


        # action here is an array with length action_horizon
        return action_pred
