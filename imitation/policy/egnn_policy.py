import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn.pool import global_mean_pool

from imitation.policy.base_policy import BasePolicy
from imitation.model.egnn import EGNN

import logging
import wandb
import os
from tqdm.auto import tqdm


log = logging.getLogger(__name__)


class EGNNPolicy(BasePolicy):
    def __init__(self,
                    dataset,
                    action_dim: int,
                    node_feature_dim: int,
                    obs_node_feature_dim: int,
                    pred_horizon: int,
                    obs_horizon: int,
                    ckpt_path=None,
                    lr: float = 1e-3,
                    batch_size=64,
                    use_normalization=True):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device {self.device}")

        self.dataset = dataset
        self.action_dim = action_dim
        self.use_normalization = use_normalization
        self.batch_size = batch_size
        self.node_feature_dim = node_feature_dim
        self.obs_node_feature_dim = obs_node_feature_dim

        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.lr = lr

        self.model = EGNN (
            in_node_nf=self.obs_node_feature_dim * self.obs_horizon,
            hidden_nf=256,
            out_node_nf=self.action_dim * self.pred_horizon,
            in_edge_nf=1,
            n_layers=5,
            normalize=True
        ).to(self.device)
        
        self.global_epoch = 0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.ckpt_path = ckpt_path
        
    def load_nets(self, ckpt_path):
        log.info(f"Loading model from {ckpt_path}")
        try:
            self.model.load_state_dict(torch.load(ckpt_path))
        except:
            log.error(f"Could not load model from {ckpt_path}")

    def save_nets(self, ckpt_path):
        log.info(f"Saving model to {ckpt_path}")
        torch.save(self.model.state_dict(), ckpt_path)

    def get_action(self, obs_deque):
        
        nobs = []
        for obs in obs_deque:
            nobs.append(obs.x)
        nobs = torch.stack(nobs, dim=1)
        
        nobs = nobs.flatten(start_dim=1)
        pred, x = self.model(h=nobs.to(self.device).float(),
                            edges=obs_deque[0].edge_index.to(self.device).long(),
                            edge_attr=obs_deque[0].edge_attr.to(self.device).unsqueeze(1).float(),
                            x=obs.pos[:,:3].to(self.device).float(),
        )

        if self.use_normalization:
            pred = self.dataset.unnormalize_data(pred, stats_key='action')
        pred = global_mean_pool(pred, batch=torch.zeros(pred.shape[0], dtype=torch.long).to(self.device))
        pred = pred.reshape(-1, self.pred_horizon, self.action_dim)
        return pred.detach().cpu().numpy() # return joint values only

    def validate(self, dataset, model_path):
        '''
        Calculate validation loss for noise prediction model in the given dataset
        '''
        loss_fn = nn.MSELoss()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            with tqdm(dataset, desc='Val Batch', leave=False) as tbatch:
                for nbatch in tbatch:
                    if self.use_normalization:
                        nbatch.y = self.dataset.normalize_data(nbatch.y, stats_key='action')
                    nobs = nbatch.x.to(self.device).float()
                    nobs = nobs.flatten(start_dim=1)
                    action = nbatch.y.to(self.device).float()
                    action = action.flatten(start_dim=1)
                    pred, x = self.model(h=nobs, 
                                        edges=nbatch.edge_index.to(self.device).long(),
                                        edge_attr=nbatch.edge_attr.to(self.device).unsqueeze(1).float(),
                                        x=nbatch.pos[:,:3].to(self.device).float(),
                    )
                    pred = global_mean_pool(pred, torch.zeros(pred.shape[0], dtype=torch.long).to(self.device))
                    loss = loss_fn(pred, action)
                    val_loss += loss.item()
        val_loss /= len(dataset)
        return val_loss
    

    def train(self, dataset, num_epochs, model_path, seed=0):
        '''Train the policy on the given dataset for the given number of epochs.
        Usinf self.model.forward() to get the action for the given observation.'''

        # set seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        loss_fn = nn.MSELoss()
        
        # visualize data in batch
        batch = next(iter(dataloader))
        log.info(f"batch.y.shape:{batch.y.shape}")
        log.info(f"batch.x.shape: {batch.x.shape}")

        with tqdm(range(num_epochs), desc="Epoch", leave=False) as pbar:
            for epoch in pbar:
                with tqdm(dataloader, desc="Batch", leave=False) as pbar:
                    for nbatch in pbar:
                        if self.use_normalization:
                            nbatch.y = self.dataset.normalize_data(nbatch.y, stats_key='action')
                        nobs = nbatch.x.to(self.device).float()
                        nobs = nobs.flatten(start_dim=1)
                        action = nbatch.y.to(self.device).float()
                        action = action.flatten(start_dim=1)

                        pred, x = self.model(h=nobs, 
                                          edges=nbatch.edge_index.to(self.device).long(),
                                          edge_attr=nbatch.edge_attr.to(self.device).unsqueeze(1).float(),
                                          x=nbatch.pos[:,:3].to(self.device).float(),
                        )
                        pred = global_mean_pool(pred, nbatch.batch.to(self.device))
                        loss = loss_fn(pred, action)
                        # loss_x = loss_fn(x, nbatch.pos[:,:3].to(self.device).float())
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        pbar.set_postfix({"loss": loss.item()})
                        wandb.log({"loss": loss.item()})    

                self.global_epoch += 1
                wandb.log({"epoch": self.global_epoch, "loss": loss.item()})

                # save model
                torch.save(self.model.state_dict(), model_path)
                pbar.set_description(f"Epoch: {self.global_epoch}, Loss: {loss.item()}")

        
        torch.save(self.model.state_dict(), model_path + f'{num_epochs}_ep.pt')
