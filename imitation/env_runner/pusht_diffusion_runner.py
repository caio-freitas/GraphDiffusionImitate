import collections
import logging
from typing import Dict
import torch
import numpy as np
import os
import gdown

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from imitation.agent.base_agent import BaseAgent
from imitation.env_runner.base_runner import BaseRunner
from imitation.dataset.pusht_state_dataset import PushTStateDataset
from imitation.model.diffusion_policy.conditional_unet1d import ConditionalUnet1D

from diffusion_policy.env.pusht.pusht_env import PushTEnv

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


class PushtDiffusionRunner(BaseRunner):
    def __init__(self, output_dir) -> None:
        super().__init__(output_dir)
        # parameters    
        self.obs_dim = 5
        self.action_dim = 2
        
        self.pred_horizon = 16
        self.obs_horizon = 2
        self.action_horizon = 8
        self.max_steps = 200 # TODO add to param file
        self.device = torch.device("cpu") # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = PushTEnv()
        self.obs = self.env.reset()
        # keep a queue of last 2 steps of observations
        self.obs_deque = collections.deque(
            [self.obs] * self.obs_horizon, maxlen=self.obs_horizon)
        
        self._init_stats()
        self.num_diffusion_iters = 100
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
        # get stats from dataset
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
        print("batch['obs'].shape:", batch['obs'].shape)
        print("batch['action'].shape", batch['action'].shape)

    def reset(self) -> None:
        self.obs = self.env.reset()

    def run(self, agent: BaseAgent, n_steps: int) -> Dict:
        log.info(f"Running agent {agent.__class__.__name__} for {n_steps} steps")
        
        done = False
        step_idx = 0
        for i in range(n_steps):
            B = 1
            # stack the last obs_horizon (2) number of observations
            obs_seq = np.stack(self.obs_deque)
            # normalize observation
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

            # only take action_horizon number of actions
            start = self.obs_horizon - 1
            end = start + self.action_horizon
            action = action_pred[start:end,:]
            # (action_horizon, action_dim)

            # self.obs = self.env.render(mode='human')
            # self.env.step(agent.act(self.obs))

            # execute action_horizon number of steps
            # without replanning
            for i in range(len(action)):
                # stepping env
                obs, reward, done, info = self.env.step(action[i])
                # save observations
                self.obs_deque.append(obs)
                # and reward/vis
                # rewards.append(reward)
                # imgs.append(env.render(mode='rgb_array'))
                self.env.render(mode='human')
                # update progress bar
                step_idx += 1
                # pbar.update(1)
                # pbar.set_postfix(reward=reward)
                if step_idx > self.max_steps:
                    done = True
                if done:
                    break
        self.env.close()

