import logging
import os

import numpy as np
import torch
import torch.nn as nn

from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import wandb

from imitation.policy.base_policy import BasePolicy

from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from torch_geometric.data import DataLoader

log = logging.getLogger(__name__)

def normalize_data(data, stats):
    for obs in data:
        obs = (obs - stats['min']) / (stats['max'] - stats['min'])
        obs = obs * 2 - 1
    return data

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data
# TODO use normalization


class DiffusionGraphPolicy(BasePolicy):
    def __init__(self, 
                    obs_dim: int,
                    action_dim: int,
                    node_feature_dim: int,
                    num_edge_types: int,
                    pred_horizon: int,
                    obs_horizon: int,
                    action_horizon: int,
                    num_diffusion_iters: int,
                    dataset: BaseLowdimDataset,
                    denoising_network: nn.Module,
                    ckpt_path= None,
                    lr: float = 1e-4,
                    batch_size: int = 256):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ckpt_path = ckpt_path
        self.node_feature_dim = node_feature_dim
        
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.num_diffusion_iters = num_diffusion_iters
        self.lr = lr

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f"Using device {self.device}")
        # create network object
        self.noise_pred_net = denoising_network

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
        

        # self._init_stats()
        self.load_nets(self.ckpt_path)

        # create dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=32,
            shuffle=True,
        )


    def load_nets(self, ckpt_path):
        if ckpt_path is None:
            log.info('No pretrained weights given.')
            self.ema_noise_pred_net = self.noise_pred_net.to(self.device)
            
        if not os.path.isfile(ckpt_path):
            log.error(f"Pretrained weights not found at {ckpt_path}. ")
            self.ema_noise_pred_net = self.noise_pred_net.to(self.device)
        try: 
            state_dict = torch.load(ckpt_path, map_location=self.device)
            self.ema_noise_pred_net = self.noise_pred_net
            self.ema_noise_pred_net.load_state_dict(state_dict)
            self.ema_noise_pred_net.to(self.device)
            log.info( 'Pretrained weights loaded.')
        except:
            log.error('Error loading pretrained weights.')
            self.ema_noise_pred_net = self.noise_pred_net.to(self.device)

    def _init_stats(self):
        # save training data statistics (min, max) for each dim
        self.stats = self.dataset.stats

        log.info(f"Dataset stats: {self.stats}")

    def get_action(self, obs_deque):
        B = 1 # action shape is (B, Ta, Da), observations (B, To, Do)
        # nobs = normalize_data(obs_seq, stats=self.stats['obs'])
        # transform deques to numpy arrays
        obs_cond = []
        pos = []
        G_t = obs_deque[-1]
        for i in range(len(obs_deque)):
            obs_cond.append(obs_deque[i].y[:,3:]) # only quaternions
            pos.append(obs_deque[i].pos)
        obs_cond = torch.cat(obs_cond, dim=1)
        obs_pos = torch.cat(pos, dim=0)
        with torch.no_grad():
            # initialize action from Guassian noise


            noisy_action = torch.randn( # +1 object
                (self.action_dim + 1, self.pred_horizon, self.node_feature_dim), device=self.device)
            naction = noisy_action

            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred, x = self.ema_noise_pred_net(
                    x = naction,
                    edge_index = G_t.edge_index,
                    edge_attr = G_t.edge_attr,
                    x_coord = G_t.y[:,:3],
                    cond = obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

        # unnormalize action
        naction = naction.detach().to('cpu').numpy()
        # 
        action = naction[:9,:,0].T
        # action_pred = unnormalize_data(naction, stats=self.stats['action'])
        
        # (action_horizon, action_dim)
        return action

    def train(self, 
              dataset=None, 
              num_epochs=100,
              model_path="last.pt",
              seed=0):
        '''
        Trains the noise prediction network, using self.dataset
        Resulting in the self.ema_noise_pred_net object.
        '''
        log.info('Training noise prediction network.')
            
        # set seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Standard ADAM optimizer
        # Note that EMA parameters are not optimized
        self.noise_pred_net.to(self.device)
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
                    for batch in tepoch:
                        # device transfer
                        nobs = batch.y.to(self.device)
                        naction = batch.x.to(self.device)
                        B = 1 # fixed to one, 

                        # observation as FiLM conditioning
                        # (B, node, obs_horizon, obs_dim)
                        obs_cond = nobs[:,:,3:] # only quaternions
                        # (B, obs_horizon * obs_dim)
                        obs_cond = obs_cond.flatten(start_dim=2)

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

                        # guarantees it to be float32
                        noisy_actions = noisy_actions.float()
                        obs_cond = obs_cond.float()       


                        # predict the noise residual
                        noise_pred, x = self.noise_pred_net(
                            noisy_actions, batch.edge_index, batch.edge_attr, x_coord = batch.y[:,-1,:3], cond=obs_cond)

                        # L2 loss
                        loss = nn.functional.mse_loss(noise_pred, noise)
                        wandb.log({'noise_pred_loss': loss})

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
                wandb.log({'epoch_loss': np.mean(epoch_loss)})
                # save model checkpoint
                torch.save(self.noise_pred_net.state_dict(), model_path)

        # Weights of the EMA model
        # is used for inference
        # ema_noise_pred_net = ema.averaged_model
        self.ema_noise_pred_net = self.noise_pred_net