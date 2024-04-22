import torch
from torch import nn
from torch_geometric.loader import DataLoader

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
                    node_feature_dim: int,
                    obs_node_feature_dim: int,
                    pred_horizon: int,
                    obs_horizon: int,
                    ckpt_path=None,
                    lr: float = 1e-3):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device {self.device}")

        self.dataset = dataset
        self.node_feature_dim = node_feature_dim
        self.obs_node_feature_dim = obs_node_feature_dim

        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.lr = lr

        self.model = EGNN (
            in_node_nf=self.obs_node_feature_dim * self.obs_horizon,
            hidden_nf=256,
            out_node_nf=self.node_feature_dim * self.pred_horizon,
            in_edge_nf=1,
            n_layers=5,
            normalize=True
        ).to(self.device)
        
        self.global_epoch = 0

        
        self.ckpt_path = ckpt_path
        
    def load_nets(self, ckpt_path):
        log.info(f"Loading model from {ckpt_path}")
        try:
            self.model.load_state_dict(torch.load(ckpt_path))
        except:
            log.error(f"Could not load model from {ckpt_path}")    

    def get_action(self, obs_deque):
        
        nobs = []
        for obs in obs_deque:
            nobs.append(obs.y)
        y = torch.stack(nobs, dim=1).to(self.device).float()
        nobs = y.flatten(start_dim=1)
        # import pdb; pdb.set_trace()
        pred, x = self.model(h=nobs, 
                            edges=obs_deque[0].edge_index.to(self.device).long(),
                            edge_attr=obs_deque[0].edge_attr.to(self.device).unsqueeze(1).float(),
                            x=obs.pos[:,:3].to(self.device).float(),
        )
        pred = pred.reshape(-1, self.pred_horizon, self.node_feature_dim)
        return pred[:9,:,0].T.detach().cpu().numpy() # return joint values only

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
                    nobs = nbatch.y.to(self.device).float()
                    nobs = nobs.flatten(start_dim=1)
                    action = nbatch.x.to(self.device).float()
                    pred, x = self.model(h=nobs, 
                                        edges=nbatch.edge_index.to(self.device).long(),
                                        edge_attr=nbatch.edge_attr.to(self.device).unsqueeze(1).float(),
                                        x=nbatch.pos[:,:3].to(self.device).float(),
                    )
                    pred = pred.reshape(-1, self.pred_horizon, self.node_feature_dim)
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
            batch_size=32,
            shuffle=True
        )

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # LR scheduler with warmup
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
        # visualize data in batch
        batch = next(iter(dataloader))
        log.info(f"batch.y.shape:{batch.y.shape}")
        log.info(f"batch.x.shape: {batch.x.shape}")

        with tqdm(range(num_epochs), desc="Epoch", leave=False) as pbar:
            for epoch in pbar:
                with tqdm(dataloader, desc="Batch", leave=False) as pbar:
                    for nbatch in pbar:
                        nobs = nbatch.y.to(self.device).float()
                        nobs = nobs.flatten(start_dim=1)
                        action = nbatch.x.to(self.device).float()

                        pred, x = self.model(h=nobs, 
                                          edges=nbatch.edge_index.to(self.device).long(),
                                          edge_attr=nbatch.edge_attr.to(self.device).unsqueeze(1).float(),
                                          x=nbatch.pos[:,:3].to(self.device).float(),
                        )
                        pred = pred.reshape(-1, self.pred_horizon, self.node_feature_dim)
                        loss = loss_fn(pred, action)
                        # loss_x = loss_fn(x, nbatch.pos[:,:3].to(self.device).float())
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_scheduler.step()

                        pbar.set_postfix({"loss": loss.item()})
                        wandb.log({"loss": loss.item()})    

                self.global_epoch += 1
                wandb.log({"epoch": self.global_epoch, "loss": loss.item()})

                # save model
                torch.save(self.model.state_dict(), model_path)
                pbar.set_description(f"Epoch: {self.global_epoch}, Loss: {loss.item()}")

        
        torch.save(self.model.state_dict(), model_path + f'{num_epochs}_ep.pt')
