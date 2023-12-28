import torch
from torch import nn

from imitation.policy.base_policy import BasePolicy
from imitation.model.mlp import MLPNet

import logging
import wandb
import os
from tqdm.auto import tqdm


log = logging.getLogger(__name__)


class MLPPolicy(BasePolicy):
    def __init__(self,
                    env,
                    model: nn.Module,
                    dataset,
                    obs_dim: int,
                    action_dim: int,
                    pred_horizon: int,
                    obs_horizon: int,
                    action_horizon: int,
                    ckpt_path=None):
        super().__init__(env)
        self.env = env # TODO remove
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.dataset = dataset
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.model = model.to(self.device)
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

        self.ckpt_path = ckpt_path
        
    def load_nets(self, ckpt_path):
        log.info(f"Loading model from {ckpt_path}")
        self.model.load_state_dict(torch.load(ckpt_path))
        

    def get_action(self, obs):
        obs = torch.tensor([obs], dtype=torch.float32).to(self.device)
        action = self.model(obs).detach().cpu().numpy()
        # reshape action to be pred_horizon x action_dim
        action = action.reshape(self.pred_horizon, self.action_dim)
        return action

    def train(self, dataset, num_epochs, model_path, seed=0):
        '''Train the policy on the given dataset for the given number of epochs.
        Usinf self.model.forward() to get the action for the given observation.'''

        # set seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        loss_fn = nn.MSELoss() # TODO change to abs loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # visualize data in batch
        batch = next(iter(self.dataloader))
        log.info(f"batch['obs'].shape:{batch['obs'].shape}")
        log.info(f"batch['action'].shape: {batch['action'].shape}")

        with tqdm(range(num_epochs)) as pbar:
            for epoch in pbar:
                for nbatch in self.dataloader:
                    nobs = nbatch['obs'].to(self.device).float()
                    nobs = nobs.flatten(start_dim=1)
                    action = nbatch['action'].to(self.device).float()
                    action = action.flatten(start_dim=1)
                        
                    pred = self.model(nobs)
                    loss = loss_fn(pred, action)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                wandb.log({"epoch": epoch, "loss": loss.item()})

                # save model
                torch.save(self.model.state_dict(), model_path)
                pbar.set_description(f"Epoch: {epoch}, Loss: {loss.item()}")

        wandb.finish()
        
        torch.save(self.model.state_dict(), model_path + f'{num_epochs}_ep.pt')
