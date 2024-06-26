import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.pool import global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops
import math

from imitation.model.egnn import E_GCL, EGNN

class MPLayer(MessagePassing):
    '''
    Custom message passing layer for the GraphARM model
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='sum') #  "Max" aggregation.
        self.f = nn.Sequential(Linear(3 * in_channels, in_channels),
                       nn.ReLU(),
                       Linear(in_channels, in_channels)) # MLP for message construction
        self.g = nn.Sequential(Linear(3 * in_channels, in_channels),
                          nn.ReLU(),
                          Linear(in_channels, in_channels)) # MLP for attention coefficients
        
        self.gru = nn.GRU(2*in_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr):
        '''
        x has shape [N, in_channels]
        edge_index has shape [2, E]
        **self-loops should be added in the preprocessing step (fully connecting the graph)
        '''

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out, _ = self.gru(torch.cat([x, out], dim=-1)) # discard final hidden state
        return out

    def message(self, x_i, x_j, edge_attr):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        h_vi = x_i
        h_vj = x_j
        h_eij = edge_attr

        m_ij = self.f(torch.cat([h_vi, h_vj, h_eij], dim=-1))
        a_ij = torch.sigmoid(self.g(torch.cat([h_vi, h_vj, h_eij], dim=-1)))
        return m_ij * a_ij


class EGraphConditionEncoder(nn.Module):
    '''
    Graph Convolutional Network (GCN) for encoding the graph-level conditioning vector
    '''
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers=3, device=None):
        super().__init__()
        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = 16
        self.MAX_EMBS = 30
        self.id_embedding = nn.Embedding(self.MAX_EMBS, self.embed_dim).to(self.device)
        self.egnn = EGNN(
            in_node_nf=input_dim + self.embed_dim, 
            out_node_nf=hidden_dim,
            hidden_nf=hidden_dim,
            in_edge_nf=1,
            n_layers=n_layers,
            normalize=False).to(self.device)
        self.pool = global_max_pool
        self.fc = Linear(hidden_dim, output_dim).to(self.device)

    def forward(self, x, edge_index, coord, edge_attr, batch=None, ids=None):
        x = x.float().to(self.device).flatten(start_dim=1)
        ids = ids.long().to(self.device)
        coord = coord.float().to(self.device)
        edge_attr = edge_attr.float().to(self.device)
        edge_index = edge_index.to(self.device)
        batch = batch.long().to(self.device) if batch is not None else torch.zeros(x.shape[0], dtype=torch.long).to(self.device)
        id_embed = self.id_embedding(ids)
        x = torch.cat([x, id_embed], dim=-1)
        h_v, x = self.egnn(x, coord, edge_index, edge_attr)
        g_v = self.pool(h_v,batch=batch)
        h_v = self.fc(g_v)
        return h_v

class ConditionalARGDenoising(nn.Module):
    def __init__(self,
                node_feature_dim,
                cond_feature_dim,
                obs_horizon,
                pred_horizon,
                edge_feature_dim,
                num_edge_types,
                num_layers=5,
                hidden_dim=256,
                device=None):
        '''
        Denoising GNN (based on GraphARM) with FiLM conditioning on 
        "global" conditioning vector (encoded observation)
        '''
        super().__init__()
        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        num_edge_types += 1
        self.num_layers = num_layers
        self.node_feature_dim = node_feature_dim
        self.cond_feature_dim = cond_feature_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.hidden_dim = hidden_dim    
        self.node_embedding = Linear(self.node_feature_dim*pred_horizon, hidden_dim).to(self.device)
        self.edge_embedding = Linear(edge_feature_dim, hidden_dim).to(self.device)
        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        self.cond_channels = hidden_dim * 2 * self.num_layers
        self.cond_encoder = EGraphConditionEncoder(
            input_dim = self.cond_feature_dim * self.obs_horizon, 
            output_dim = self.cond_channels, 
            hidden_dim = hidden_dim, 
            device=self.device
        )


        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(E_GCL(
                input_nf=hidden_dim,
                output_nf=hidden_dim,
                hidden_nf=hidden_dim,
                edges_in_d=hidden_dim,
                normalize=True # helps in stability / generalization
            ).to(self.device))
        
        self.node_pred_layer = nn.Sequential(Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, self.node_feature_dim*self.pred_horizon)
        ).to(self.device)
        
    def forward(self, x, edge_index, edge_attr, x_coord, film_cond, batch=None):
        # make sure x and edge_attr are of type float, for the MLPs
        x = x.float().to(self.device).flatten(start_dim=1)
        edge_attr = edge_attr.float().to(self.device).unsqueeze(-1) # add channel dimension
        edge_index = edge_index.to(self.device)
        film_cond = film_cond.float().to(self.device)
        x_coord = x_coord.float().to(self.device)
        batch = batch.long().to(self.device) if batch is not None else torch.zeros(x.shape[0], dtype=torch.long).to(self.device)
        

        assert x.shape[0] == x_coord.shape[0], "x and x_coord must have the same length"

        h_v = self.node_embedding(x)
        h_e = self.edge_embedding(edge_attr.reshape(-1, 1))

        
        embed = film_cond.reshape(self.num_layers, 2, self.hidden_dim)
        scales = embed[:,0,...]
        biases = embed[:,1,...]
        x_v = x_coord
        # instead of convolution, run message passing
        for l in range(self.num_layers):
            # FiLM conditioning
            h_v = scales[l] * h_v + biases[l]
            h_v, x_v, edge_attr_pred = self.layers[l](h_v, edge_index, coord=x_v, edge_attr=h_e)
        
        # graph-level embedding, from average pooling layer
        graph_embedding = global_mean_pool(h_v, batch=batch)

        # repeat graph embedding to have the same shape as h_v
        graph_embedding = graph_embedding[batch]

        node_pred = self.node_pred_layer(torch.cat([graph_embedding, h_v], dim=1)) # 2*hidden_dim
        
        node_pred = node_pred.reshape(-1, self.pred_horizon, self.node_feature_dim) # reshape to original shape        

        return node_pred, x_v

class ConditionalGraphNoisePred(nn.Module):
    def __init__(self,
                node_feature_dim,
                cond_feature_dim,
                obs_horizon,
                pred_horizon,
                edge_feature_dim,
                num_edge_types,
                num_layers=5,
                hidden_dim=256,
                diffusion_step_embed_dim=32,
                num_diffusion_steps=200,
                device=None):
        '''
        Denoising EGNN with FiLM conditioning on 
        graph-level conditioning vector (encoded observation)
        '''
        super().__init__()
        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        num_edge_types += 1
        self.num_layers = num_layers
        self.node_feature_dim = node_feature_dim
        self.cond_feature_dim = cond_feature_dim # 6D rotation only, since positions are x_coord
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.hidden_dim = hidden_dim    
        self.diffusion_step_embed_dim = diffusion_step_embed_dim
        self.num_diffusion_steps = num_diffusion_steps # TODO parameterize
        self.node_embedding = Linear(self.node_feature_dim*pred_horizon, hidden_dim).to(self.device)
        self.edge_embedding = Linear(edge_feature_dim, hidden_dim).to(self.device)
        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        self.cond_channels = (hidden_dim + self.diffusion_step_embed_dim) * 2 * self.num_layers
        self.cond_encoder = EGraphConditionEncoder(
            input_dim = self.cond_feature_dim * self.obs_horizon, 
            output_dim = self.cond_channels, 
            hidden_dim = hidden_dim, 
            device=self.device
        )


        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(E_GCL(
                input_nf=hidden_dim + self.diffusion_step_embed_dim,
                output_nf=hidden_dim + self.diffusion_step_embed_dim,
                hidden_nf=hidden_dim,
                edges_in_d=hidden_dim,
                normalize=True # helps in stability / generalization
            ).to(self.device))
        
        self.node_pred_layer = nn.Sequential(Linear(hidden_dim + self.diffusion_step_embed_dim , hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, self.node_feature_dim*self.pred_horizon)
        ).to(self.device)
        
        self.diffusion_step_encoder = nn.Sequential(
            Linear(self.diffusion_step_embed_dim, self.diffusion_step_embed_dim * 4),
            nn.Mish(),
            Linear(self.diffusion_step_embed_dim * 4, self.diffusion_step_embed_dim)
        ).to(self.device)

        self.FILL_VALUE = 0.0
        self.pe = self.positionalencoding(self.num_diffusion_steps)

    def positionalencoding(self, lengths):
        '''
        From Chen, et al. 2021 (Order Matters: Probabilistic Modeling of Node Sequences for Graph Generation)
        * lengths: length(s) of graph in the batch
        '''
        l_t = lengths # .max() # use when parallelizing
        pes = torch.zeros([l_t, self.diffusion_step_embed_dim], device=self.device)
        position = torch.arange(0, l_t, device=self.device).unsqueeze(1) + 1
        div_term = torch.exp((torch.arange(0, self.diffusion_step_embed_dim, 2, dtype=torch.float, device=self.device) *
                              -(math.log(10000.0) / self.diffusion_step_embed_dim)))
        pes[:,0::2] = torch.sin(position.float() * div_term)
        pes[:,1::2] = torch.cos(position.float() * div_term)
        return pes


    def forward(self, x, edge_index, edge_attr, x_coord, cond, timesteps, batch=None):
        # make sure x and edge_attr are of type float, for the MLPs
        x = x.float().to(self.device).flatten(start_dim=1)
        edge_attr = edge_attr.float().to(self.device).unsqueeze(-1) # add channel dimension
        edge_index = edge_index.to(self.device)
        ids = cond[:,0,-1].long().to(self.device)
        cond = cond[:,:,:-1].float().to(self.device)
        x_coord = x_coord.float().to(self.device)
        timesteps = timesteps.to(self.device)
        batch = batch.long().to(self.device)
        batch_size = batch[-1] + 1

        timesteps_embed = self.diffusion_step_encoder(self.pe[timesteps])
        timesteps_embed = timesteps_embed[batch]

        assert x.shape[0] == x_coord.shape[0], "x and x_coord must have the same length"

        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=x.shape[0], fill_value=self.FILL_VALUE)

        h_v = self.node_embedding(x)

        h_v = torch.cat([h_v, timesteps_embed], dim=-1)

        h_e = self.edge_embedding(edge_attr.reshape(-1, 1))

        # FiLM generator
        embed = self.cond_encoder(cond, edge_index, x_coord, edge_attr, batch=batch, ids=ids)
        embed = embed.reshape(self.num_layers, batch_size, 2, (self.hidden_dim + self.diffusion_step_embed_dim))
        scales = embed[:,:,0,...]
        biases = embed[:,:,1,...]
        x_v = x_coord
        # instead of convolution, run message passing
        for l in range(self.num_layers):
            # FiLM conditioning
            h_v = scales[l,batch] * h_v + biases[l,batch]
            h_v, x_v, edge_attr_pred = self.layers[l](h_v, edge_index, coord=x_v, edge_attr=h_e)

        node_pred = self.node_pred_layer(h_v) # hidden_dim
        
        node_pred = node_pred.reshape(-1, self.pred_horizon, self.node_feature_dim) # reshape to original shape        

        return node_pred, x_v
    
