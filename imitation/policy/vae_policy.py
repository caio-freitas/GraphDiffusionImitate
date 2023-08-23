import torch
from torch import nn

from imitation.policy.base_policy import BasePolicy
from imitation.model.mlp import MLPNet
from imitation.dataset.pusht_state_dataset import PushTStateDataset

import logging
import wandb
import os
from tqdm.auto import tqdm

os.environ["WANDB_DISABLED"] = "false"


log = logging.getLogger(__name__)



class VAEPolicy(BasePolicy):
    def __init__(self,
                    env,
                    model: nn.Module,
                    action_dim: int,
                    dataset = PushTStateDataset(
                        dataset_path='./pusht_cchi_v7_replay.zarr.zip',
                        pred_horizon=1,
                        obs_horizon=1,
                        action_horizon=1,
                    ),
                    ckpt_path=None):
        super().__init__(env)
        self.env = env
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.dataset = dataset
        self.pred_horizon = self.dataset.pred_horizon
        self.action_dim = action_dim
        self.model = model
        log.info(f"Model: {self.model}")
        # load model from ckpt
        if ckpt_path is not None:
            self.load_nets(ckpt_path)
        # create dataloader
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=4,
            num_workers=1,
            shuffle=False, # tp overfit
            # accelerate cpu-gpu transfer
            pin_memory=True,
            # don't kill worker process afte each epoch
            persistent_workers=True,
            collate_fn=self.collate_fn
        )

        self.ckpt_path = ckpt_path
        
    def collate_fn(self, batch):
        return {
            'action': torch.stack([x['action'] for x in batch]),
            'obs': torch.stack([x['obs'] for x in batch])
        }


    def load_nets(self, ckpt_path):
        log.info(f"Loading model from {ckpt_path}")
        self.model.load_state_dict(torch.load(ckpt_path))


    def get_action(self, obs, latent=None):
        ''' 
        Implement sampling for VAE Policy
        obs: torch.Tensor of shape (obs_dim) (currently not used)
        latent: torch.Tensor of shape (1, latent_dim) 
        to choose a specific latent vector
        '''
        if latent is None:
            latent = torch.randn(1, self.model.latent_dim).to(self.device)
        else:
            latent = torch.tensor(latent)
            latent = latent.type(torch.FloatTensor)
            latent = latent.to(self.device)
            # print(f"latent: {latent}")
        action = self.model.decode(latent)
        action = torch.reshape(action, (self.pred_horizon, self.action_dim)) # TODO remove gripper dim
        return action.detach().cpu().numpy()

    def get_latent_space_dist(self):
        '''Get latent space distribution from dataset'''
        # set dataloader batch size to 1
        # self.dataloader.batch_size = 1
        latents = []
        with tqdm(self.dataloader) as pbar:
            for nbatch in pbar:
                action = nbatch['action'].to(self.device).float()
                # get first action
                action = action[0]
                # print(f"action: {action}")
                latent = self.model.encode(action)
                # print(f"latent: {latent}")
                latents.append(latent[0].detach().cpu().numpy())

        return latents


    def elbo_loss(self, x, x_hat, mean, logvar):
        '''Implement ELBO loss for VAE Policy'''
        # is binary cross entropy the right distribution?
        # reconstruction_loss = nn.functional.cross_entropy(x_hat, x, reduction='mean')
        reconstruction_loss = nn.functional.mse_loss(x_hat, x)

        KLD = - 0.5 * torch.sum(1+ logvar - mean.pow(2) - logvar.exp())
        wandb.log({"KLD": KLD.item(), "reconstruction_loss": reconstruction_loss.item()})
        return reconstruction_loss + KLD

    def train(self, dataset, num_epochs, model_path):
        '''Train the Variation Autoencoder Model on the given dataset for the given number of epochs.
        '''

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # visualize data in batch
        batch = next(iter(self.dataloader))
        log.info(f"batch['obs'].shape:{batch['obs'].shape}")
        log.info(f"batch['action'].shape: {batch['action'].shape}")

        # keep first self.action_dim

        with tqdm(range(num_epochs)) as pbar:
            for epoch in pbar:
                for nbatch in self.dataloader:
                    action = nbatch['action'].to(self.device).float()
                    action = action.flatten(start_dim=1)
                    x_hat, mean, log_var = self.model(action)
                    loss = self.elbo_loss(action, x_hat, mean, log_var)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                wandb.log({"epoch": epoch, "loss": loss.item()})

                log.info(f'Epoch: {epoch}, Loss: {loss.item()}')
                # save model
                torch.save(self.model.state_dict(), model_path)
                pbar.set_description(f"Epoch: {epoch}, Loss: {loss.item()}")

        wandb.finish()

        torch.save(self.model.state_dict(), model_path + f'{num_epochs}_ep.pt')
