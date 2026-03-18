"""
GDDPM: Graph-based Denoising Diffusion Probabilistic Model
Adapted from: https://github.com/AmirMiraki/GDDPM

Original paper:
  AmirMiraki et al. "Probabilistic forecasting of renewable energy and electricity demand
  using Graph-based Denoising Diffusion Probabilistic Model", Energy and AI, 2024.
  https://doi.org/10.1016/j.egyai.2024.100459

This file adapts the GDDPM EpsilonTheta denoising network to the robot-graph diffusion
setting used in this project.  The key architectural differences from the project's existing
ConditionalGraphNoisePred are:

  1. Temporal backbone: dilated 1-D convolutions (vs. EGNN message-passing).
  2. Spatial backbone: GatedGraphConv (torch_geometric) inside each residual block,
     applied jointly over the dilated-conv output.
  3. Conditioning: GraphCondEncoder encodes graph-structured observations into a
     per-graph conditioning vector using GatedGraphConv (same spatial operator as
     the residual blocks); upsampled with CondUpsampler MLP before injection into
     every residual block (instead of FiLM scales/biases per EGNN layer).

"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GatedGraphConv
from torch_geometric.utils import add_self_loops
from torch_geometric.nn.pool import global_mean_pool



# ---------------------------------------------------------------------------
# Diffusion step embedding  (Section 3.2 of the GDDPM paper)
# ---------------------------------------------------------------------------

class DiffusionEmbedding(nn.Module):
    """
    Sinusoidal embedding for the diffusion timestep, then projected through
    two linear layers.  Adapted from epsilon_theta.py in the GDDPM repo.
    """
    def __init__(self, dim: int, proj_dim: int, max_steps: int = 500):
        super().__init__()
        self.register_buffer("embedding",
                             self._build_embedding(dim, max_steps),
                             persistent=False)
        self.projection1 = nn.Linear(dim * 2, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step: torch.Tensor) -> torch.Tensor:
        x = self.embedding[diffusion_step]          # (B, dim*2)
        x = F.silu(self.projection1(x))             # (B, proj_dim)
        x = F.silu(self.projection2(x))             # (B, proj_dim)
        return x

    @staticmethod
    def _build_embedding(dim: int, max_steps: int) -> torch.Tensor:
        steps = torch.arange(max_steps).unsqueeze(1)            # [T, 1]
        dims  = torch.arange(dim).unsqueeze(0)                  # [1, dim]
        table = steps * 10.0 ** (dims * 4.0 / dim)             # [T, dim]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table                                             # [T, dim*2]


# ---------------------------------------------------------------------------
# Conditioning upsampler MLP  (lightweight replacement for RNN encoder in GDDPM)
# ---------------------------------------------------------------------------

class CondUpsampler(nn.Module):
    """
    Two-layer MLP that projects the global graph conditioning vector to target_dim,
    the number of nodes (or a spatial dim used by the residual blocks).
    """
    def __init__(self, cond_length: int, target_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(cond_length, target_dim // 2)
        self.linear2 = nn.Linear(target_dim // 2, target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.linear1(x), 0.4)
        x = F.leaky_relu(self.linear2(x), 0.4)
        return x


# ---------------------------------------------------------------------------
# Graph conditioning encoder  (replaces EGraphConditionEncoder from graph_diffusion.py)
# ---------------------------------------------------------------------------

class GraphCondEncoder(nn.Module):
    """
    GNN encoder for graph-structured observations, aligned with the GDDPM paper.

    Uses GatedGraphConv (same spatial operator as the ResidualBlocks) rather
    than the E(N)-equivariant EGNN from EGraphConditionEncoder. No coordinate
    updates, no equivariance overhead.

    Architecture:
      1. Flatten temporal obs: (N, obs_horizon, F) -> (N, obs_horizon*F)
      2. Linear input projection -> (N, hidden_dim)
      3. GatedGraphConv message passing (n_layers iterations, weight-shared)
      4. Global mean pooling per graph -> (B, hidden_dim)
      5. Linear output projection -> (B, output_dim)

    Args:
        input_dim:  obs_horizon * cond_feature_dim (flattened obs per node)
        hidden_dim: internal feature dimension
        output_dim: output conditioning vector size
        n_layers:   GatedGraphConv iterations (default 3)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=3):
        super().__init__()
        self.input_proj   = nn.Linear(input_dim, hidden_dim)
        self.gnn          = GatedGraphConv(hidden_dim, num_layers=n_layers)
        self.output_proj  = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        """
        Args:
            x:          (N, obs_horizon, cond_feature_dim)
            edge_index: (2, E) — with self-loops already added by caller
            batch:      (N,) — node-to-graph mapping
        Returns:
            (B, output_dim) — graph-level conditioning vector
        """
        h = x.float().flatten(start_dim=1)             # (N, input_dim)
        h = F.leaky_relu(self.input_proj(h), 0.4)      # (N, hidden_dim)
        h = self.gnn(h, edge_index)                    # (N, hidden_dim)
        g = global_mean_pool(h, batch=batch)           # (B, hidden_dim)
        return self.output_proj(g)                     # (B, output_dim)


# ---------------------------------------------------------------------------
# Residual block: dilated conv  +  GatedGraphConv  +  conditioner
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Core GDDPM residual block, adapted for per-node action sequences.

    Inputs (per node, batched):
      x            - (N, residual_channels, pred_horizon) current node representation
      conditioner  - (N, 1, target_dim) upsampled global conditioning
      diffusion_step - (N, residual_channels) projected diffusion embedding
      edge_index   - graph connectivity
      edge_weight  - optional edge weights for GatedGraphConv

    The spatial GatedGraphConv is applied on x reshaped to (N, channels) per
    time-step, averaged across the time axis before mixing back in.
    """
    def __init__(self,
                 hidden_size: int,
                 residual_channels: int,
                 dilation: int):
        super().__init__()
        self.residual_channels = residual_channels

        # temporal: dilated causal conv  (C -> 2C for gated activation)
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )

        # spatial: GatedGraphConv operates on C features (time-averaged),
        # then expanded to 2C via a linear layer before mixing
        self.graph_conv    = GatedGraphConv(residual_channels, num_layers=1)
        self.graph_expand  = nn.Linear(residual_channels, 2 * residual_channels)

        # diffusion step projection: hidden_size -> 2C  (added before gating)
        self.diffusion_projection = nn.Linear(hidden_size, 2 * residual_channels)
        # conditioner: (N,1,T_up) -> (N, 2C, T_up)
        self.conditioner_projection = nn.Conv1d(
            1, 2 * residual_channels, kernel_size=1, padding=2, padding_mode="circular"
        )
        # output: C -> 2C  (residual + skip)
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self,
                x: torch.Tensor,
                conditioner: torch.Tensor,
                diffusion_step: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None) -> tuple:
        """
        x:              (N, residual_channels, T)
        conditioner:    (N, 1, T_cond)  - output of CondUpsampler unsqueezed
        diffusion_step: (N, hidden_size)
        Returns: (residual_out, skip_connection)  both (N, residual_channels, T)
        """
        N, C, T = x.shape

        # --- conditioner projection (N,1,T_up) -> (N, 2C, T'); trim to T
        cond_proj = self.conditioner_projection(conditioner)          # (N, 2C, T')
        min_cond  = min(cond_proj.shape[-1], T)
        cond_proj = cond_proj[..., :min_cond]                         # (N, 2C, T_trim)

        # --- diffusion step: (N, 2C) -> broadcast over time
        diff_proj = self.diffusion_projection(diffusion_step)         # (N, 2C)
        diff_proj = diff_proj.unsqueeze(-1)                           # (N, 2C, 1)

        # --- temporal dilated conv (N, C, T) -> (N, 2C, T')
        y_temporal = self.dilated_conv(x)                             # (N, 2C, T')
        T_conv = y_temporal.shape[-1]

        # --- spatial GatedGraphConv on time-averaged features
        x_flat   = x.mean(dim=-1)                                     # (N, C)
        y_graph  = self.graph_conv(x_flat, edge_index)                # (N, C)
        y_graph  = self.graph_expand(y_graph)                         # (N, 2C)
        y_spatial = y_graph.unsqueeze(-1).expand(N, 2 * C, T_conv)   # (N, 2C, T')

        # --- combine: align all to T_conv
        T_min = min(T_conv, min_cond)
        y = (y_temporal[..., :T_min]
             + y_spatial[..., :T_min]
             + cond_proj[..., :T_min]
             + diff_proj.expand(N, 2 * C, T_min))                     # (N, 2C, T_min)

        # --- gated activation
        gate, filt = torch.chunk(y, 2, dim=1)                         # each (N, C, T_min)
        y = torch.sigmoid(gate) * torch.tanh(filt)                    # (N, C, T_min)

        # --- output: (N, C, T_min) -> (N, 2C, T_min)
        y = F.leaky_relu(self.output_projection(y), 0.4)
        residual, skip = torch.chunk(y, 2, dim=1)                     # each (N, C, T_min)

        # --- residual skip: align back to original T
        T_res = min(residual.shape[-1], T)
        residual_out = (x[..., :T_res] + residual[..., :T_res]) / math.sqrt(2.0)
        return residual_out, skip[..., :T_res]


# ---------------------------------------------------------------------------
# Top-level GDDPM noise predictor
# ---------------------------------------------------------------------------

class GDDPMNoisePred(nn.Module):
    """
    GDDPM denoising network with the same interface as ConditionalGraphNoisePred.

    Architecture summary:
      - GraphCondEncoder encodes the graph-structured observation into a
        per-graph conditioning vector using GatedGraphConv.
      - CondUpsampler projects it to a node-dimension matching the graph size.
      - A stack of ResidualBlocks (dilated conv + GatedGraphConv) predicts noise.

    Args:
        node_feature_dim:       feature dimension per node per step (e.g. 1 for joint value)
        cond_feature_dim:       obs feature dim, e.g. 9 for joint_pos+gripper_qpos features
        obs_horizon:            number of observation steps for conditioning
        pred_horizon:           number of prediction steps (action horizon)
        edge_feature_dim:       edge attribute size (usually 1)
        num_edge_types:         number of edge type categories
        residual_layers:        number of ResidualBlock layers
        residual_channels:      channels inside each block
        dilation_cycle_length:  dilation doubles every this many layers
        hidden_dim:             hidden size for GraphCondEncoder and diffusion embed
        diffusion_step_embed_dim: raw sinusoidal embedding size (≤ hidden_dim)
        num_diffusion_steps:    total DDPM timesteps (for embedding table)
        device:                 torch device (auto-detected if None)
    """

    def __init__(self,
                 node_feature_dim: int,
                 cond_feature_dim: int,
                 obs_horizon: int,
                 pred_horizon: int,
                 edge_feature_dim: int,
                 num_edge_types: int,
                 residual_layers: int = 8,
                 residual_channels: int = 8,
                 dilation_cycle_length: int = 2,
                 hidden_dim: int = 256,
                 diffusion_step_embed_dim: int = 64,
                 num_diffusion_steps: int = 100,
                 device=None):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.node_feature_dim = node_feature_dim
        self.cond_feature_dim = cond_feature_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.hidden_dim = hidden_dim
        self.residual_channels = residual_channels
        self.num_diffusion_steps = num_diffusion_steps

        # --- Observation encoder (same as ConditionalGraphNoisePred) ----------
        # cond_channels: output length from EGraphConditionEncoder.
        # We use it as the `cond_length` fed into CondUpsampler.
        self.cond_channels = hidden_dim
        self.cond_encoder = GraphCondEncoder(
            input_dim=cond_feature_dim * obs_horizon,
            hidden_dim=hidden_dim,
            output_dim=self.cond_channels,
        ).to(self.device)

        # --- Diffusion step embedding ------------------------------------------
        self.diffusion_embedding = DiffusionEmbedding(
            dim=diffusion_step_embed_dim,
            proj_dim=hidden_dim,
            max_steps=num_diffusion_steps,
        ).to(self.device)

        # --- Conditioning upsampler ---------------------------------------------
        # Output size = pred_horizon so that it can be broadcast per time step.
        self.cond_upsampler = CondUpsampler(
            cond_length=self.cond_channels,
            target_dim=pred_horizon,
        ).to(self.device)

        # --- Input projection: (node_feature_dim) -> residual_channels ----------
        self.input_projection = nn.Conv1d(
            node_feature_dim,
            residual_channels,
            kernel_size=1,
            padding=2,
            padding_mode="circular",
        ).to(self.device)

        # --- Residual stack ------------------------------------------------------
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(
                hidden_size=hidden_dim,
                residual_channels=residual_channels,
                dilation=2 ** (i % dilation_cycle_length),
            )
            for i in range(residual_layers)
        ])
        self.residual_blocks.to(self.device)

        # --- Output projection ---------------------------------------------------
        self.skip_projection = nn.Conv1d(
            residual_channels, residual_channels, kernel_size=3
        ).to(self.device)
        self.output_projection = nn.Conv1d(
            residual_channels, node_feature_dim, kernel_size=3
        ).to(self.device)

        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    # ------------------------------------------------------------------
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                x_coord: torch.Tensor,
                cond: torch.Tensor,
                timesteps: torch.Tensor,
                batch: torch.Tensor = None):
        """
        Drop-in equivalent of ConditionalGraphNoisePred.forward.

        Args:
            x:          (N_total, pred_horizon, node_feature_dim)  noisy action
            edge_index: (2, E)
            edge_attr:  (E,) or (E, 1)  — edge attributes / types
            x_coord:    (N_total, 3) — 3D node positions
            cond:       (N_total, obs_horizon, cond_feature_dim)  — obs features
            timesteps:  (B,)  — diffusion timestep per graph in batch
            batch:      (N_total,)  — maps each node to its graph index

        Returns:
            noise_pred: (N_total, pred_horizon, node_feature_dim)
            x_coord:    (N_total, 3)  unchanged (kept for API compatibility)
        """
        # ---- move to device / cast ----------------------------------------
        x          = x.float().to(self.device)          # (N_act, T, F)
        edge_attr  = edge_attr.float().to(self.device)
        edge_index = edge_index.to(self.device)
        x_coord    = x_coord.float().to(self.device)
        timesteps  = timesteps.to(self.device)
        cond       = cond.float().to(self.device)
        # obs_batch maps OBS nodes to graphs; x may have fewer nodes per graph
        if batch is None:
            obs_batch = torch.zeros(cond.shape[0], dtype=torch.long, device=self.device)
        else:
            obs_batch = batch.long().to(self.device)

        B       = obs_batch.max().item() + 1
        obs_npg = cond.shape[0] // B   # obs nodes per graph (e.g. 10)
        act_npg = x.shape[0] // B      # action nodes per graph (e.g. 9)

        # action_batch: maps action nodes to their graph index
        action_batch = torch.arange(B, dtype=torch.long, device=self.device).repeat_interleave(act_npg)

        # ---- obs edge_index with self-loops (for EGraphConditionEncoder) ----
        edge_attr_1d = edge_attr.reshape(-1)
        obs_edge_index_sl, obs_edge_attr_sl = add_self_loops(
            edge_index, edge_attr_1d,
            num_nodes=cond.shape[0], fill_value=0.0
        )

        # ---- action edge_index: filter to robot-only edges, remap indices ---
        if act_npg < obs_npg:
            src, dst = edge_index
            src_local = src % obs_npg
            dst_local = dst % obs_npg
            mask = (src_local < act_npg) & (dst_local < act_npg)
            act_ei = edge_index[:, mask]
            act_ea = edge_attr_1d[mask]
            graph_ids_ei = act_ei[0] // obs_npg
            act_ei = act_ei % obs_npg + graph_ids_ei * act_npg
        else:
            act_ei = edge_index
            act_ea = edge_attr_1d
        act_edge_index_sl, act_edge_attr_sl = add_self_loops(
            act_ei, act_ea,
            num_nodes=x.shape[0], fill_value=0.0
        )

        # ---- Graph-level conditioning vector --------------------------------
        graph_cond = self.cond_encoder(
            cond.float().to(self.device), obs_edge_index_sl,
            batch=obs_batch,
        )                                                   # (B, cond_channels)

        # ---- Up-sample conditioning to pred_horizon -------------------------
        cond_up = self.cond_upsampler(graph_cond)          # (B, pred_horizon)

        # Broadcast from per-graph to per-ACTION-node
        cond_up_node = cond_up[action_batch]               # (N_act, pred_horizon)
        cond_up_node = cond_up_node.unsqueeze(1)           # (N_act, 1, pred_horizon)

        # ---- Diffusion step embedding ----------------------------------------
        diffusion_step = self.diffusion_embedding(timesteps)   # (B, hidden_dim)
        diffusion_step_node = diffusion_step[action_batch]     # (N_act, hidden_dim)

        # ---- Reshape x: (N_act, T, F) -> (N_act, F, T) for Conv1d ----------
        x_conv = x.permute(0, 2, 1)                           # (N_act, F, T)
        h = F.leaky_relu(self.input_projection(x_conv), 0.4)  # (N_act, C, T')

        # ---- Residual stack --------------------------------------------------
        skip_sum = None
        for block in self.residual_blocks:
            h, skip = block(
                h,
                cond_up_node,
                diffusion_step_node,
                act_edge_index_sl,
                edge_weight=None,
            )
            if skip_sum is None:
                skip_sum = skip
            else:
                # align lengths
                min_T = min(skip_sum.shape[-1], skip.shape[-1])
                skip_sum = (skip_sum[..., :min_T] + skip[..., :min_T])

        n_layers = len(self.residual_blocks)
        skip_sum = skip_sum / math.sqrt(n_layers)              # (N, C, T')

        # ---- Output projection -----------------------------------------------
        out = F.leaky_relu(self.skip_projection(skip_sum), 0.4)  # (N, C, T'')
        out = self.output_projection(out)                         # (N, F, T''')

        # ---- Crop / pad to exactly pred_horizon ------------------------------
        T_out = out.shape[-1]
        if T_out >= self.pred_horizon:
            out = out[..., :self.pred_horizon]
        else:
            # pad with zeros if output is too short (edge case)
            pad = torch.zeros(
                out.shape[0], out.shape[1], self.pred_horizon - T_out,
                device=self.device
            )
            out = torch.cat([out, pad], dim=-1)

        # (N, F, T) -> (N, T, F)
        noise_pred = out.permute(0, 2, 1)

        return noise_pred, x_coord


# ---------------------------------------------------------------------------
# Flat residual block: dilated conv only, no GatedGraphConv
# ---------------------------------------------------------------------------

class FlatResidualBlock(nn.Module):
    """
    Dilated-conv residual block for flat (non-graph) action sequences.

    Operates on batch-level tensors (B, C, T) instead of node-level (N_total, C, T).
    Same gated activation as ResidualBlock, without the GatedGraphConv branch.
    """
    def __init__(self,
                 hidden_size: int,
                 residual_channels: int,
                 dilation: int):
        super().__init__()
        self.residual_channels = residual_channels

        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )
        self.diffusion_projection = nn.Linear(hidden_size, 2 * residual_channels)
        self.conditioner_projection = nn.Conv1d(
            1, 2 * residual_channels, kernel_size=1, padding=2, padding_mode="circular"
        )
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self,
                x: torch.Tensor,
                conditioner: torch.Tensor,
                diffusion_step: torch.Tensor) -> tuple:
        """
        x:              (B, residual_channels, T)
        conditioner:    (B, 1, pred_horizon)
        diffusion_step: (B, hidden_size)
        Returns: (residual_out, skip) both (B, residual_channels, T_min)
        """
        B, C, T = x.shape

        cond_proj = self.conditioner_projection(conditioner)    # (B, 2C, T')
        min_cond  = min(cond_proj.shape[-1], T)
        cond_proj = cond_proj[..., :min_cond]

        diff_proj = self.diffusion_projection(diffusion_step)   # (B, 2C)
        diff_proj = diff_proj.unsqueeze(-1)                     # (B, 2C, 1)

        y_temporal = self.dilated_conv(x)                       # (B, 2C, T')
        T_conv = y_temporal.shape[-1]

        T_min = min(T_conv, min_cond)
        y = (y_temporal[..., :T_min]
             + cond_proj[..., :T_min]
             + diff_proj.expand(B, 2 * C, T_min))              # (B, 2C, T_min)

        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)             # (B, C, T_min)

        y = F.leaky_relu(self.output_projection(y), 0.4)
        residual, skip = torch.chunk(y, 2, dim=1)              # each (B, C, T_min)

        T_res = min(residual.shape[-1], T)
        residual_out = (x[..., :T_res] + residual[..., :T_res]) / math.sqrt(2.0)
        return residual_out, skip[..., :T_res]


# ---------------------------------------------------------------------------
# Flat GDDPM noise predictor: graph obs encoding + flat action denoising
# ---------------------------------------------------------------------------

class FlatGDDPMNoisePred(nn.Module):
    """
    Drop-in replacement for GDDPMNoisePred where the action is a flat
    (B, pred_horizon, action_dim) tensor instead of per-node.

    The graph structure is used *only* for observation encoding via
    GraphCondEncoder.  The residual denoising blocks operate on the
    full action batch at graph granularity (B, ...).

    Args:
        action_dim:             flat action dimensionality (e.g. 7 for OSC_POSE)
        cond_feature_dim:       obs feature dim, e.g. 9 for joint_pos+gripper_qpos features
        obs_horizon:            number of observation steps for conditioning
        pred_horizon:           number of prediction steps
        edge_feature_dim:       edge attribute size (usually 1)
        num_edge_types:         number of edge type categories
        residual_layers:        number of FlatResidualBlock layers
        residual_channels:      channels inside each block
        dilation_cycle_length:  dilation doubles every this many layers
        hidden_dim:             hidden size for GraphCondEncoder
        diffusion_step_embed_dim: sinusoidal embedding size
        num_diffusion_steps:    total DDPM timesteps
    """

    def __init__(self,
                 action_dim: int,
                 cond_feature_dim: int,
                 obs_horizon: int,
                 pred_horizon: int,
                 edge_feature_dim: int,
                 num_edge_types: int,
                 residual_layers: int = 8,
                 residual_channels: int = 32,
                 dilation_cycle_length: int = 2,
                 hidden_dim: int = 256,
                 diffusion_step_embed_dim: int = 64,
                 num_diffusion_steps: int = 100,
                 device=None):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.hidden_dim = hidden_dim
        self.residual_channels = residual_channels

        self.cond_channels = hidden_dim
        self.cond_encoder = GraphCondEncoder(
            input_dim=cond_feature_dim * obs_horizon,
            hidden_dim=hidden_dim,
            output_dim=self.cond_channels,
        ).to(self.device)

        self.diffusion_embedding = DiffusionEmbedding(
            dim=diffusion_step_embed_dim,
            proj_dim=hidden_dim,
            max_steps=num_diffusion_steps,
        ).to(self.device)

        self.cond_upsampler = CondUpsampler(
            cond_length=self.cond_channels,
            target_dim=pred_horizon,
        ).to(self.device)

        self.input_projection = nn.Conv1d(
            action_dim,
            residual_channels,
            kernel_size=1,
            padding=2,
            padding_mode="circular",
        ).to(self.device)

        self.residual_blocks = nn.ModuleList([
            FlatResidualBlock(
                hidden_size=hidden_dim,
                residual_channels=residual_channels,
                dilation=2 ** (i % dilation_cycle_length),
            )
            for i in range(residual_layers)
        ])
        self.residual_blocks.to(self.device)

        self.skip_projection = nn.Conv1d(
            residual_channels, residual_channels, kernel_size=3
        ).to(self.device)
        self.output_projection = nn.Conv1d(
            residual_channels, action_dim, kernel_size=3
        ).to(self.device)

        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                x_coord: torch.Tensor,
                cond: torch.Tensor,
                timesteps: torch.Tensor,
                batch: torch.Tensor = None):
        """
        Args:
            x:          (B, pred_horizon, action_dim)   flat noisy action
            edge_index: (2, E)
            edge_attr:  (E,) or (E, 1)
            x_coord:    (N_total, 3)
            cond:       (N_total, obs_horizon, cond_feature_dim)  graph obs features
            timesteps:  (B,)
            batch:      (N_total,)  node-to-graph mapping for GraphCondEncoder

        Returns:
            noise_pred: (B, pred_horizon, action_dim)
            x_coord:    (N_total, 3) unchanged
        """
        x          = x.float().to(self.device)
        edge_attr  = edge_attr.float().to(self.device)
        edge_index = edge_index.to(self.device)
        x_coord    = x_coord.float().to(self.device)
        timesteps  = timesteps.to(self.device)
        if batch is None:
            batch = torch.zeros(x_coord.shape[0], dtype=torch.long, device=self.device)
        else:
            batch = batch.long().to(self.device)

        B          = timesteps.shape[0]

        edge_attr_1d = edge_attr.reshape(-1)
        edge_index_sl, edge_attr_sl = add_self_loops(
            edge_index, edge_attr_1d,
            num_nodes=x_coord.shape[0], fill_value=0.0
        )

        # Graph-level conditioning: (B, cond_channels)
        graph_cond = self.cond_encoder(
            cond.float().to(self.device), edge_index_sl,
            batch=batch,
        )

        # Up-sample conditioning to pred_horizon: (B, pred_horizon) -> (B, 1, pred_horizon)
        cond_up = self.cond_upsampler(graph_cond).unsqueeze(1)

        # Diffusion step embedding: (B, hidden_dim)
        diffusion_step = self.diffusion_embedding(timesteps)

        # x: (B, T, Da) -> (B, Da, T) for Conv1d
        x_conv = x.permute(0, 2, 1)
        h = F.leaky_relu(self.input_projection(x_conv), 0.4)

        skip_sum = None
        for block in self.residual_blocks:
            h, skip = block(h, cond_up, diffusion_step)
            if skip_sum is None:
                skip_sum = skip
            else:
                min_T = min(skip_sum.shape[-1], skip.shape[-1])
                skip_sum = (skip_sum[..., :min_T] + skip[..., :min_T])

        n_layers = len(self.residual_blocks)
        skip_sum = skip_sum / math.sqrt(n_layers)

        out = F.leaky_relu(self.skip_projection(skip_sum), 0.4)
        out = self.output_projection(out)                       # (B, Da, T''')

        T_out = out.shape[-1]
        if T_out >= self.pred_horizon:
            out = out[..., :self.pred_horizon]
        else:
            pad = torch.zeros(
                out.shape[0], out.shape[1], self.pred_horizon - T_out,
                device=self.device
            )
            out = torch.cat([out, pad], dim=-1)

        noise_pred = out.permute(0, 2, 1)                      # (B, T, Da)
        return noise_pred, x_coord
