import logging
import os
import numpy as np
from tqdm.auto import tqdm
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler

from imitation.model.mlp import MLPNet
from imitation.policy.base_policy import BasePolicy

from diffusion_policy.dataset.base_dataset import BaseLowdimDataset

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



class MLPDiffusionUnet1DPolicy(BasePolicy):
    def __init__(self, 
                    obs_dim: int,
                    action_dim: int,
                    pred_horizon: int,
                    obs_horizon: int,
                    action_horizon: int,
                    num_diffusion_iters: int,
                    dataset: BaseLowdimDataset,
                    ckpt_path: str,
                    lr: float = 1e-4):
        super().__init__()
        self.dataset = dataset
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ckpt_path = ckpt_path
        
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.num_diffusion_iters = num_diffusion_iters
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device {self.device}")
        # create network object
        self.noise_pred_net = MLPNet(
            input_dim=self.action_dim + self.obs_dim,
            output_dim=self.action_dim,
            hidden_dims=[256, 64, 4, 2, 64],
            activation=torch.nn.LeakyReLU(),
            output_activation=torch.nn.Identity()
        )


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
        

        self._init_stats()

    def load_nets(self, ckpt_path):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Pretrained weights not found at {ckpt_path}. ")

        state_dict = torch.load(ckpt_path, map_location='cuda')
        self.ema_noise_pred_net = self.noise_pred_net
        self.ema_noise_pred_net.load_state_dict(state_dict)
        self.ema_noise_pred_net.to(self.device)
        log.info( 'Pretrained weights loaded.')

    def _init_stats(self):
        # save training data statistics (min, max) for each dim
        self.stats = self.dataset.stats

        # create dataloader
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=256,
            num_workers=1,
            shuffle=True,
            # accelerate cpu-gpu transfer
            pin_memory=True,
            # don't kill worker process afte each epoch
            persistent_workers=True
        )

        # visualize data in batch
        batch = next(iter(self.dataloader))
        log.info(f"batch['obs'].shape:{batch['obs'].shape}")
        log.info(f"batch['action'].shape: {batch['action'].shape}")

    def get_action(self, obs_seq):
        B = 1 # action shape is (B, Ta, Da), observations (B, To, Do)
        nobs = normalize_data([obs_seq], stats=self.stats['obs'])
        # device transfer
        nobs = torch.from_numpy(nobs).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, self.pred_horizon, self.action_dim), device=self.device)
            naction = noisy_action

            action_obs = torch.cat([naction, nobs], dim=-1)

            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            for k in self.noise_scheduler.timesteps:
                # predict noise

                noise_pred = self.ema_noise_pred_net(action_obs)

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

    def train(self, 
              dataset=None, 
              num_epochs=100,
              model_path="./mlp_diffusion_last.pt"):
        '''
        Trains the noise prediction network, using self.dataset
        Resulting in the self.ema_noise_pred_net object.
        '''
        log.info('Training noise prediction network.')
            
        # Standard ADAM optimizer
        optimizer = torch.optim.AdamW(
            params=self.noise_pred_net.parameters(),
            lr=self.lr, weight_decay=1e-6)

        # Cosine LR schedule with linear warmup
        lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=len(self.dataloader) * num_epochs
        )

        with tqdm(range(num_epochs), desc='Epoch') as tglobal:
            # epoch loop
            for epoch_idx in tglobal:
                epoch_loss = list()
                # batch loop
                with tqdm(self.dataloader, desc='Batch', leave=False) as tepoch:
                    for nbatch in tepoch:
                        # data normalized in dataset
                        # device transfer
                        nobs = nbatch['obs'].to(self.device)
                        # obs_cond = nobs.flatten(start_dim=1)
                        naction = nbatch['action'].to(self.device)
                        B = nobs.shape[0]

                        # sample noise to add to actions
                        noise = torch.randn(naction.shape, device=self.device)

                        # sample a diffusion iteration for each data point
                        timesteps = torch.randint(
                            0, self.noise_scheduler.config.num_train_timesteps,
                            (B,), device=self.device
                        ).long()
                        # add noise to the clean images according to the noise magnitude at each diffusion iteration
                        # (this is the forward diffusion process)
                        noisy_actions = self.noise_scheduler.add_noise(
                            naction, noise, timesteps)
                        
                        # predict the noise residual
                        noise_pred = self.noise_pred_net(
                            torch.cat([noisy_actions, nobs], dim=-1))
                        # L2 loss
                        loss = torch.nn.functional.mse_loss(noise_pred, noise)

                        # optimize
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                        # step lr scheduler every batch
                        # this is different from standard pytorch behavior
                        lr_scheduler.step() 

                        # logging
                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)
                tglobal.set_postfix(loss=np.mean(epoch_loss))
                # save model checkpoint
                torch.save(self.noise_pred_net.state_dict(), model_path)

        # Weights of the EMA model
        # is used for inference
        # ema_noise_pred_net = ema.averaged_model
        self.ema_noise_pred_net = self.noise_pred_net