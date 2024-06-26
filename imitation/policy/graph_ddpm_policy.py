import logging
import os

import numpy as np
import torch
import torch.nn as nn

from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
import wandb

from imitation.policy.base_policy import BasePolicy

from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from torch_geometric.data import DataLoader

log = logging.getLogger(__name__)


class GraphConditionalDDPMPolicy(BasePolicy):
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
                    batch_size: int = 256,
                    use_normalization: bool = True,
                    keep_first_action: bool = True,):
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
        self.use_normalization = use_normalization
        self.keep_first_action = keep_first_action
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f"Using device {self.device}")
        # create network object
        self.noise_pred_net = denoising_network.to(self.device)
        self.ema_noise_pred_net = self.noise_pred_net.to(self.device)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon',
            beta_start=1e-4,
            beta_end=2e-2,
        )
        
        self.lr_scheduler = None
        self.optimizer = None
        self.num_epochs = None

        self.global_epoch = 0
        self.last_naction = torch.zeros((self.action_dim, self.pred_horizon, self.node_feature_dim), device=self.device)
        self.playback_count = 0

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

    def save_nets(self, ckpt_path):
        torch.save(self.ema_noise_pred_net.state_dict(), ckpt_path)
        log.info(f"Model saved at {ckpt_path}")

    def MOCK_get_graph_from_obs(self): # for testing purposes, remove before merge
        # plays back observation from dataset
        playback_graph = self.dataset[self.playback_count]
        obs_cond    = playback_graph.y
        playback_graph.x = playback_graph.x[:,0,:]
        self.playback_count += 7
        log.info(f"Playing back observation {self.playback_count}")
        return obs_cond, playback_graph
    
    def get_action(self, obs_deque):
        B = 1 # action shape is (B, Ta, Da), observations (B, To, Do)
        # transform deques to numpy arrays
        obs_cond = []
        pos = []
        G_t = obs_deque[-1]
        for i in range(len(obs_deque)):
            obs_cond.append(obs_deque[i].y.unsqueeze(1))
            pos.append(obs_deque[i].pos)
        obs = torch.cat(obs_cond, dim=1)
        obs_pos = torch.cat(pos, dim=0)
        if self.use_normalization:
            nobs = self.dataset.normalize_data(obs, stats_key='obs')
            nobs[:,:,-1] = obs[:,:,-1] # skip normalization for node IDs
            self.last_naction = self.dataset.normalize_data(G_t.x.unsqueeze(1), stats_key='action').to(self.device)
        else:
            self.last_naction = G_t.x.unsqueeze(1).to(self.device)

        with torch.no_grad():
            # initialize action from Guassian noise
            noisy_action = torch.randn((G_t.x.shape[0], self.pred_horizon, self.node_feature_dim), device=self.device)
            naction = noisy_action

            if self.keep_first_action:
                naction[:,0,:] = self.last_naction[:,-1,:1] 

            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred, x = self.ema_noise_pred_net(
                    x = naction,
                    edge_index = G_t.edge_index,
                    edge_attr = G_t.edge_attr,
                    x_coord = G_t.pos[:,:3],    
                    cond = nobs,
                    timesteps = torch.tensor([k], dtype=torch.long, device=self.device),
                    batch = torch.zeros(naction.shape[0], dtype=torch.long, device=self.device)
                )
                
                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

                if self.keep_first_action:
                    naction[:,0,:] = self.last_naction[:,-1,:1]


        # add node dimension, to pass through normalizer
        naction = torch.cat([naction, torch.zeros((naction.shape[0], self.pred_horizon, 1), device=self.device)], dim=2)
        naction = naction.detach().to('cpu')
        if self.use_normalization:
            naction = self.dataset.unnormalize_data(naction, stats_key='action').numpy()
        action = naction[:self.action_dim,:,0].T
        
        # (action_horizon, action_dim)
        return action

    def validate(self, dataset=None, model_path="last.pt"):
        '''
        Validates the noise prediction network, using self.dataset
        '''
        log.info('Validating noise prediction network.')
        # create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        self.load_nets(model_path)
        self.ema_noise_pred_net.eval()
        with torch.no_grad():
            val_loss = list()
            for batch in dataloader:
                nobs = batch.y
                if self.use_normalization:
                    # normalize observation
                    nobs = self.dataset.normalize_data(batch.y, stats_key='obs').to(self.device)
                    nobs[:,:,-1] = batch.y[:,:,-1] # skip normalization for node IDs
                    # normalize action
                    naction = self.dataset.normalize_data(batch.x, stats_key='action').to(self.device)
                B = batch.num_graphs

                # observation as FiLM conditioning
                # (B, node, obs_horizon, obs_dim)
                obs_cond = nobs
                naction = naction[:,:,:1] # joint value
                # (B, obs_horizon * obs_dim)

                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (B,), device=self.device
                ).long()

                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)

                # split naction into (B, N_nodes, pred_horizon, node_feature_dim), selecting the items from each batch.batch
                naction = torch.cat([naction[batch.batch == i].unsqueeze(0) for i in batch.batch.unique()], dim=0)

                # sample noise to add to actions

                noise = torch.randn(naction.shape, device=self.device, dtype=torch.float32)

                noisy_actions = self.noise_scheduler.add_noise(
                    naction, noise, timesteps)
                
                if self.keep_first_action:
                    noisy_actions[:,:,0,:] = naction[:,:,0,:]

                # guarantees it to be float32
                noisy_actions = noisy_actions.float()
                obs_cond = obs_cond.float()

                # stack the batch dimension
                noisy_actions = noisy_actions.flatten(end_dim=1)
                # stack noise in the batch dimension
                noise = noise.flatten(end_dim=1)

                # predict the noise residual
                noise_pred, x = self.ema_noise_pred_net(
                    noisy_actions, 
                    batch.edge_index, 
                    batch.edge_attr, 
                    x_coord = batch.pos[:,:3], 
                    cond=obs_cond,
                    timesteps=timesteps,
                    batch=batch.batch)
                
                # L2 loss
                loss = nn.functional.mse_loss(noise_pred, noise)
                val_loss.append(loss.item())
        return np.mean(val_loss)
    

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

        if self.num_epochs is None:
            log.warn(f"Global num_epochs not set. Using {num_epochs}.")
            self.num_epochs = num_epochs
        
        # set seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        ema = EMAModel(
            parameters=self.ema_noise_pred_net.parameters(),
            power=0.75)

        # Standard ADAM optimizer
        # Note that EMA parameters are not optimized
        self.noise_pred_net.to(self.device)
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                params=self.noise_pred_net.parameters(),
                lr=self.lr, 
                weight_decay=1e-6,
                betas=[0.95, 0.999],
                eps=1e-8)
            
        if self.lr_scheduler is None:
        # Cosine LR schedule with linear warmup
            self.lr_scheduler = get_scheduler(
                name='cosine',
                optimizer=self.optimizer,
                num_warmup_steps=500,
                num_training_steps=len(dataloader) * self.num_epochs
            )

        with tqdm(range(num_epochs), desc='Epoch') as tglobal:
            # epoch loop
            for epoch_idx in tglobal:
                epoch_loss = list()
                # batch loop
                with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                    for batch in tepoch:
                        nobs = batch.y
                        if self.use_normalization:
                            # normalize observation
                            nobs = self.dataset.normalize_data(batch.y, stats_key='obs').to(self.device)
                            nobs[:,:,-1] = batch.y[:,:,-1] # skip normalization for node IDs
                            # normalize action
                            naction = self.dataset.normalize_data(batch.x, stats_key='action').to(self.device)
                        B = batch.num_graphs

                        # observation as FiLM conditioning
                        # (B, node, obs_horizon, obs_dim)
                        obs_cond = nobs
                        naction = naction[:,:,:1] # joint value

                        # sample a diffusion iteration for each data point
                        timesteps = torch.randint(
                            0, self.noise_scheduler.config.num_train_timesteps,
                            (B,), device=self.device
                        ).long()

                        # add noise to the clean images according to the noise magnitude at each diffusion iteration
                        # (this is the forward diffusion process)
                        # split naction into (B, N_nodes, pred_horizon, node_feature_dim), selecting the items from each batch.batch

                        naction = torch.cat([naction[batch.batch == i].unsqueeze(0) for i in batch.batch.unique()], dim=0)

                        # add noise to first action instead of sampling from Gaussian
                        noise = torch.randn(naction.shape, device=self.device, dtype=torch.float32)

                        noisy_actions = self.noise_scheduler.add_noise(
                            naction, noise, timesteps)

                        if self.keep_first_action:
                            noisy_actions[:,:,0,:] = naction[:,:,0,:]


                        # guarantees it to be float32
                        noisy_actions = noisy_actions.float()
                        obs_cond = obs_cond.float()       

                        # stack the batch dimension
                        noisy_actions = noisy_actions.flatten(end_dim=1)
                        # stack noise in the batch dimension
                        noise = noise.flatten(end_dim=1)

                        # predict the noise residual
                        noise_pred, x = self.noise_pred_net(
                            noisy_actions, 
                            batch.edge_index, 
                            batch.edge_attr, 
                            x_coord = batch.pos[:,:3], 
                            cond=obs_cond,
                            timesteps=timesteps,
                            batch=batch.batch)

                        # L2 loss
                        loss = nn.functional.mse_loss(noise_pred, noise)
                        wandb.log({'noise_pred_loss': loss, 'lr': self.lr_scheduler.get_last_lr()[0]})

                        # optimize
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        # step lr scheduler every batch
                        # this is different from standard pytorch behavior
                        self.lr_scheduler.step()

                        ema.step(self.noise_pred_net.parameters())


                        # logging
                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)
                tglobal.set_postfix(loss=np.mean(epoch_loss))
                wandb.log({'epoch': self.global_epoch, 'epoch_loss': np.mean(epoch_loss)})
                # save model checkpoint
                # use weights of the EMA model for inference
                self.save_nets(model_path)
                self.global_epoch += 1
                tglobal.set_description(f"Epoch: {self.global_epoch}")

        