'''
Diffusion Policy for imitation learning with graphs
'''
import torch
from tqdm import tqdm
import torch.nn as nn

from imitation.policy.graph_diffusion_policy import DiffusionOrderingNetwork, DenoisingNetwork
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler


class GraphDiffusionPolicy(nn.Module):
    def __init__(self,
                 dataset,
                 node_feature_dim,
                 num_edge_types,
                 denoising_network,
                 diffusion_ordering_network,
                 device='cpu'):
        super(GraphDiffusionPolicy, self).__init__()
        self.dataset = dataset
        self.device = device
        self.node_feature_dim = node_feature_dim
        self.num_edge_types = num_edge_types
        self.denoising_network = denoising_network.to(self.device)
        self.diffusion_ordering_network = diffusion_ordering_network.to(self.device)
        
        # Noise scheduler
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

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=1,
            shuffle=True,
            # accelerate cpu-gpu transfer
            pin_memory=True,
            # don't kill worker process afte each epoch
            persistent_workers=True
        )

    def compute_loss(self, data):
        '''
        Compute loss for training
        '''


    def train(self, data, num_epochs=100):
        '''
        Train noise prediction model
        '''
        self.optimizer.zero_grad()
        # Cosine LR schedule with linear warmup
        lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=self.optimizer,
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
                        # nobs = nbatch['obs'].to(self.device)
                        naction = nbatch['action'].to(self.device)
                        B = naction.shape[0]

                        # observation as FiLM conditioning
                        # (B, obs_horizon, obs_dim)
                        # obs_cond = nobs[:,:self.obs_horizon,:]
                        # # (B, obs_horizon * obs_dim)
                        # obs_cond = obs_cond.flatten(start_dim=1)


                        # sample a diffusion iteration for each data point
                        timesteps = torch.randint(
                            0, self.noise_scheduler.config.num_train_timesteps,
                            (B,), device=self.device
                        ).long()

                        # sample noise to add to actions
                        noise = torch.randn(naction.shape, device=self.device)

                        noisy_actions = self.noise_scheduler.add_noise(
                            naction, noise, timesteps)
                        

                        # guarantees it to be float32
                        noisy_actions = noisy_actions.float()
                        # obs_cond = obs_cond.float()       

                        # predict the noise residual
                        new_node, new_connections = self.denoising_network(
                            noisy_actions, timesteps)

                        # L2 loss
                        loss = nn.functional.mse_loss(new_node, new_original_nodde)

                        # optimize
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        # step lr scheduler every batch
                        # this is different from standard pytorch behavior
                        lr_scheduler.step()

                        # TODO use EMA
                        # update Exponential Moving Average of the model weights
                        # ema.step(noise_pred_net)


                        # logging
                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)