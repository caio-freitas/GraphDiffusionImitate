import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
import wandb

from imitation.policy.base_policy import BasePolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from torch_geometric.data import DataLoader

log = logging.getLogger(__name__)


def compute_snr(timesteps: torch.Tensor, noise_scheduler) -> torch.Tensor:
    """Compute signal-to-noise ratio at each timestep."""
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
    alpha_t = sqrt_alphas_cumprod[timesteps]
    sigma_t = sqrt_one_minus_alphas_cumprod[timesteps]
    snr = (alpha_t / sigma_t) ** 2
    return snr


class OSCGraphDDPMPolicy(BasePolicy):
    """
    DDPM policy for OSC_POSE control.

    The graph is used *only* for observation encoding.  The action space is a
    flat (B, pred_horizon, action_dim) tensor (e.g. 7-D EEF vector for OSC_POSE)
    with no graph structure.

    Differences from GraphConditionalDDPMPolicy:
    - No `node_feature_dim` — actions are flat, not per-node.
    - `last_naction` shape is (1, pred_horizon, action_dim).
    - Training reshapes batch.y from (action_dim*B, pred_horizon, 1) to
      (B, pred_horizon, action_dim) before computing diffusion loss.
    - Inference initialises noise as (1, pred_horizon, action_dim).
    """

    def __init__(self,
                 action_dim: int,
                 num_edge_types: int,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 num_diffusion_iters: int,
                 dataset: BaseLowdimDataset,
                 denoising_network: nn.Module,
                 ckpt_path=None,
                 lr: float = 1e-4,
                 batch_size: int = 256,
                 use_normalization: bool = True,
                 keep_first_action: bool = True,
                 num_warmup_steps: int = 100):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.ckpt_path = ckpt_path

        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.num_diffusion_iters = num_diffusion_iters
        self.lr = lr
        self.use_normalization = use_normalization
        self.keep_first_action = keep_first_action
        self.num_warmup_steps = num_warmup_steps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f"Using device {self.device}")

        self.noise_pred_net = denoising_network.to(self.device)
        self.ema_noise_pred_net = self.noise_pred_net.to(self.device)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon',
            beta_start=1e-4,
            beta_end=2e-2,
        )

        self.lr_scheduler = None
        self.optimizer = None
        self.num_epochs = None

        self.global_epoch = 0
        # (1, pred_horizon, action_dim)
        self.last_naction = torch.zeros(
            (1, self.pred_horizon, self.action_dim), device=self.device
        )
        self.playback_count = 0

    def reset(self):
        """Reset stateful inference buffers between episodes."""
        self.last_naction = torch.zeros(
            (1, self.pred_horizon, self.action_dim), device=self.device
        )
        self.playback_count = 0

    def load_nets(self, ckpt_path):
        if ckpt_path is None:
            log.info('No pretrained weights given.')
            self.ema_noise_pred_net = self.noise_pred_net.to(self.device)
            return
        if not os.path.isfile(ckpt_path):
            log.error(f"Pretrained weights not found at {ckpt_path}.")
            self.ema_noise_pred_net = self.noise_pred_net.to(self.device)
            return
        try:
            state_dict = torch.load(ckpt_path, map_location=self.device)
            self.ema_noise_pred_net = self.noise_pred_net
            self.ema_noise_pred_net.load_state_dict(state_dict)
            self.ema_noise_pred_net.to(self.device)
            log.info('Pretrained weights loaded.')
        except Exception:
            log.error('Error loading pretrained weights.')
            self.ema_noise_pred_net = self.noise_pred_net.to(self.device)

    def save_nets(self, ckpt_path):
        torch.save(self.ema_noise_pred_net.state_dict(), ckpt_path)
        log.info(f"Model saved at {ckpt_path}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_action(self, obs_deque):
        """
        obs_deque: deque of PyG Data objects (length == obs_horizon).
        Returns: action (action_horizon, action_dim) numpy array.
        """
        obs_cond = []
        G_t = obs_deque[-1]
        for i in range(len(obs_deque)):
            obs_cond.append(obs_deque[i].x.unsqueeze(1))
        obs = torch.cat(obs_cond, dim=1)   # (N_nodes, obs_horizon, feat)

        if self.use_normalization:
            nobs = self.dataset.normalize_data(obs, stats_key='obs')
            nobs[:, :, -1] = obs[:, :, -1]   # preserve node IDs
        else:
            nobs = obs

        with torch.no_grad():
            noisy_action = torch.randn(
                (1, self.pred_horizon, self.action_dim), device=self.device
            )

            if self.keep_first_action:
                noisy_action[:, 0, :] = self.last_naction[:, -1, :]

            batch_idx = torch.zeros(
                G_t.x.shape[0], dtype=torch.long, device=self.device
            )

            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            for k in self.noise_scheduler.timesteps:
                noise_pred, _ = self.ema_noise_pred_net(
                    x=noisy_action,
                    edge_index=G_t.edge_index,
                    edge_attr=G_t.edge_attr,
                    x_coord=G_t.pos[:, :3],
                    cond=nobs,
                    timesteps=torch.tensor([k], dtype=torch.long, device=self.device),
                    batch=batch_idx,
                )
                noisy_action = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action,
                ).prev_sample

                if self.keep_first_action:
                    noisy_action[:, 0, :] = self.last_naction[:, -1, :]

        naction = noisy_action.detach().cpu()   # (1, pred_horizon, action_dim)
        self.last_naction = naction

        if self.use_normalization:
            # naction[0] is (pred_horizon, action_dim) — matches per-dim normalizer
            # (scale shape (action_dim,)) which expects last dim = action_dim
            action = self.dataset.unnormalize_data(
                naction[0], stats_key='action'
            )  # (pred_horizon, action_dim)
            action = action[:self.pred_horizon, :].numpy()
        else:
            action = naction[0, :self.pred_horizon, :].numpy()

        return action

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, dataset=None, model_path="last.pt"):
        log.info('Validating noise prediction network.')
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.load_nets(model_path)
        self.ema_noise_pred_net.eval()

        with torch.no_grad():
            val_loss = []
            for batch in dataloader:
                B = batch.num_graphs
                nobs = batch.x
                # batch.y: (action_dim*B, pred_horizon, 1) from PyG DataLoader
                # Reshape to (B, pred_horizon, action_dim) BEFORE normalizing so
                # the per-dim normalizer (scale shape (action_dim,)) sees last dim=action_dim
                action_raw = batch.y.view(B, self.action_dim, self.pred_horizon, 1)
                action_raw = action_raw[:, :, :, 0].permute(0, 2, 1)  # (B, T, Da)
                if self.use_normalization:
                    nobs = self.dataset.normalize_data(batch.x, stats_key='obs').to(self.device)
                    nobs[:, :, -1] = batch.x[:, :, -1]
                    naction = self.dataset.normalize_data(action_raw, stats_key='action').to(self.device)
                else:
                    naction = action_raw.to(self.device)

                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (B,), device=self.device
                ).long()

                noise = torch.randn_like(naction)
                noisy_actions = self.noise_scheduler.add_noise(naction, noise, timesteps)

                if self.keep_first_action:
                    noisy_actions[:, 0, :] = naction[:, 0, :]

                obs_cond = nobs.float()
                noisy_actions = noisy_actions.float()

                with torch.cuda.amp.autocast():
                    noise_pred, _ = self.ema_noise_pred_net(
                        noisy_actions,
                        batch.edge_index,
                        batch.edge_attr,
                        x_coord=batch.pos[:, :3],
                        cond=obs_cond,
                        timesteps=timesteps,
                        batch=batch.batch,
                    )
                    # Per-sample MSE loss
                    loss_per_sample = F.mse_loss(noise_pred, noise, reduction='none').mean(dim=(1, 2))  # (B,)
                    # Min-SNR-5 reweighting
                    snr = compute_snr(timesteps, self.noise_scheduler)
                    min_snr_weight = torch.clamp(snr, max=5.0) / snr
                    loss = (min_snr_weight * loss_per_sample).mean()
                val_loss.append(loss.item())

        return np.mean(val_loss)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self,
              dataset=None,
              num_epochs=100,
              model_path="last.pt",
              seed=0):
        log.info('Training noise prediction network.')

        if self.num_epochs is None:
            log.warning(f"Global num_epochs not set. Using {num_epochs}.")
            self.num_epochs = num_epochs

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        ema = EMAModel(parameters=self.ema_noise_pred_net.parameters(), power=0.75)

        self.noise_pred_net.to(self.device)
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                params=self.noise_pred_net.parameters(),
                lr=self.lr,
                weight_decay=1e-6,
                betas=[0.95, 0.999],
                eps=1e-8,
            )

        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                name='cosine',
                optimizer=self.optimizer,
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=len(dataloader) * self.num_epochs,
            )

        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

        with tqdm(range(num_epochs), desc='Epoch') as tglobal:
            for _ in tglobal:
                epoch_loss = []
                with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                    for batch in tepoch:
                        B = batch.num_graphs

                        nobs = batch.x
                        # batch.y: (action_dim*B, pred_horizon, 1) from PyG DataLoader
                        # Reshape to (B, pred_horizon, action_dim) BEFORE normalizing so
                        # the per-dim normalizer (scale shape (action_dim,)) sees last dim=action_dim
                        action_raw = batch.y.view(B, self.action_dim, self.pred_horizon, 1)
                        action_raw = action_raw[:, :, :, 0].permute(0, 2, 1)  # (B, T, Da)
                        if self.use_normalization:
                            nobs = self.dataset.normalize_data(batch.x, stats_key='obs').to(self.device)
                            nobs[:, :, -1] = batch.x[:, :, -1]
                            naction = self.dataset.normalize_data(action_raw, stats_key='action').to(self.device)
                        else:
                            naction = action_raw.to(self.device)

                        timesteps = torch.randint(
                            0, self.noise_scheduler.config.num_train_timesteps,
                            (B,), device=self.device
                        ).long()

                        noise = torch.randn_like(naction)
                        noisy_actions = self.noise_scheduler.add_noise(naction, noise, timesteps)

                        if self.keep_first_action:
                            noisy_actions[:, 0, :] = naction[:, 0, :]

                        noisy_actions = noisy_actions.float()
                        obs_cond = nobs.float()

                        with torch.cuda.amp.autocast():
                            noise_pred, _ = self.noise_pred_net(
                                noisy_actions,
                                batch.edge_index,
                                batch.edge_attr,
                                x_coord=batch.pos[:, :3],
                                cond=obs_cond,
                                timesteps=timesteps,
                                batch=batch.batch,
                            )

                            # Per-sample MSE loss
                            loss_per_sample = F.mse_loss(noise_pred, noise, reduction='none').mean(dim=(1, 2))  # (B,)
                            # Min-SNR-5 reweighting
                            snr = compute_snr(timesteps, self.noise_scheduler)
                            min_snr_weight = torch.clamp(snr, max=5.0) / snr
                            loss = (min_snr_weight * loss_per_sample).mean()

                        wandb.log({
                            'noise_pred_loss': loss,
                            'lr': self.lr_scheduler.get_last_lr()[0],
                            'min_snr_weight_mean': min_snr_weight.mean().item(),
                        })

                        if scaler is not None:
                            scaler.scale(loss).backward()
                            scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.noise_pred_net.parameters(), max_norm=1.0)
                            scaler.step(self.optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.noise_pred_net.parameters(), max_norm=1.0)
                            self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.lr_scheduler.step()
                        ema.step(self.noise_pred_net.parameters())

                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)

                tglobal.set_postfix(loss=np.mean(epoch_loss))
                wandb.log({'epoch': self.global_epoch, 'epoch_loss': np.mean(epoch_loss)})
                self.save_nets(model_path)
                self.global_epoch += 1
                tglobal.set_description(f"Epoch: {self.global_epoch}")
