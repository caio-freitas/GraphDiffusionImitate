import torch
from torch import nn

from imitation.policy.base_policy import BasePolicy
from imitation.model.mlp import MLPNet
from imitation.dataset.pusht_state_dataset import PushTStateDataset

import logging
import wandb
import os

os.environ["WANDB_DISABLED"] = "true"


log = logging.getLogger(__name__)



class MLPPolicy(BasePolicy):
    def __init__(self,
                    env,
                    model: nn.Module,
                    dataset = PushTStateDataset(
                        dataset_path='./pusht_cchi_v7_replay.zarr.zip',
                        pred_horizon=1,
                        obs_horizon=1,
                        action_horizon=1,
                    ),
                    ckpt_path=None):
        super().__init__(env)
        self.env = env
        self.dataset = dataset
        self.model = model
        # load model from ckpt
        if ckpt_path is not None:
            self.load_nets(ckpt_path)
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

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        self.ckpt_path = ckpt_path
        
        
    def load_nets(self, ckpt_path):
        self.model.load_state_dict(torch.load(ckpt_path))


    def get_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        return self.model.forward(obs).detach().cpu().numpy()

    def train(self, dataset, num_epochs, model_path):
        '''Train the policy on the given dataset for the given number of epochs.
        Usinf self.model.forward() to get the action for the given observation.'''

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)


        wandb.init(
            project="mlp_policy",
            # track hyperparameters and run metadata
            config={
            "architecture": "MLP",
            "n_epochs": num_epochs,
            "lr": optimizer.param_groups[0]['lr'],
            "hidden_dims": self.model.hidden_dims,
            "activation": self.model.activation,
            "output_activation": self.model.output_activation,
            "loss": loss_fn,
            "episodes": len(self.dataset.indices),
            "batch_size": self.dataloader.batch_size,
            "env": "PushT",
            }
        )

        # visualize data in batch
        batch = next(iter(self.dataloader))
        log.info(f"batch['obs'].shape:{batch['obs'].shape}")
        log.info(f"batch['action'].shape: {batch['action'].shape}")

        for epoch in range(num_epochs):
            for nbatch in self.dataloader:
                nobs = nbatch['obs'].to(self.device)
                action = nbatch['action'].to(self.device)
                pred = self.model(nobs)
                loss = loss_fn(pred, action)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            wandb.log({"epoch": epoch, "loss": loss.item()})

            log.info(f'Epoch: {epoch}, Loss: {loss.item()}')

            # save model
            torch.save(self.model.state_dict(), model_path)

        wandb.finish()
        
        torch.save(self.model.state_dict(), model_path + f'{num_epochs}_ep.pt')
