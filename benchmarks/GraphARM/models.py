import torch
from torch import nn
from torch_geometric.nn import GAT
from torch_geometric.utils import add_self_loops, degree
from torch.nn import functional as F
import math

class DiffusionOrderingNetwork(nn.Module):
    '''
    at each diffusion step t, we sample from this network to select a node 
    v_sigma(t) to be absorbed and obtain the corresponding masked graph Gt
    '''
    def __init__(self,
                 node_feature_dim,
                 num_node_types,
                 num_edge_types,
                 num_layers=3,
                 out_channels=1,
                 num_heads=6):
        super(DiffusionOrderingNetwork, self).__init__()

        # add positional encodings into node features
        self.embedding = nn.Embedding(num_embeddings=num_node_types, embedding_dim=node_feature_dim)

        self.gat = GAT(
            in_channels=node_feature_dim,
            out_channels=node_feature_dim,
            hidden_channels=num_heads * 6,
            num_layers=num_layers,
            dropout=0,
            heads=num_heads,
        )


    def positionalencoding(self, lengths, permutations):
        '''
        From Chen, et al. 2021 (Order Matters: Probabilistic Modeling of Node Sequences for Graph Generation)
        '''
        # length = sum([len(perm) for perm in permutations])
        l_t = len(permutations[0])
        # pes = [torch.zeros(length, self.d_model) for length in lengths]
        pes = torch.split(torch.zeros((sum(lengths), self.d_model), device=self.device), lengths)
        position = torch.arange(0, l_t, device=self.device).unsqueeze(1) + 1
        div_term = torch.exp((torch.arange(0, self.d_model, 2, dtype=torch.float, device=self.device) *
                              -(math.log(10000.0) / self.d_model)))
        # test = torch.sin(position.float() * div_term)
        for i in range(len(lengths)):
            pes[i][permutations[i], 0::2] = torch.sin(position.float() * div_term)
            pes[i][permutations[i], 1::2] = torch.cos(position.float() * div_term)

        pes = torch.cat(pes)
        return pes

    def forward(self, G, p=None):
        h = self.gat(G.x.float(), G.edge_index.long())

        # TODO augment node features with positional encodings
        # if p is not None:
        #     # p = self.positionalencoding(G.batch_num_nodes().tolist(), p) original from Chen et al.
        #     p = self.positionalencoding(G.x.shape[0], p)
        #     h = h + p
        
        # softmax over nodes
        h = F.softmax(h, dim=0)
        
        return h # outputs probabilities for a categorical distribution over nodes
    
    


class DenoisingNetwork(nn.Module):
    def __init__(self,
                 node_feature_dim,
                 num_node_types,
                 num_edge_types,
                 num_layers,
                 out_channels,
                 num_heads=8):
        super(DenoisingNetwork, self).__init__()

        self.embedding_layer = nn.Embedding(num_embeddings=num_node_types, embedding_dim=node_feature_dim)
        
        self.gat = GAT(
            in_channels=node_feature_dim,
            out_channels=node_feature_dim,
            hidden_channels=num_heads * 16,
            num_layers=num_layers,
            dropout=0,
            heads=num_heads,
        )

        # Node type prediction
        self.node_type_prediction = nn.Linear(node_feature_dim, num_node_types) # Use only element of the new node
        
        # Edge type prediction
        self.edge_type_prediction = nn.Linear(node_feature_dim, num_edge_types) # Use all elements (connections to other nodes)

    def forward(self, data):

        '''
        Outputs: 
        new_node_type: type of new node to be unmasked
        new_edge_type: types of new edges from previous nodes to the one to be unmasked
        '''

        h = self.embedding_layer(data.x.squeeze().long())

        h = self.gat(h, data.edge_index.long())

        # TODO check if default attention mechanism is used

        # Node type prediction
        node_type_logits = self.node_type_prediction(h)
        # Applying softmax for the multinomial distribution
        node_type_probs = F.softmax(node_type_logits, dim=-1)

        # Edge type prediction
        edge_type_logits = self.edge_type_prediction(h)
        # Applying softmax for the multinomial distribution
        edge_type_probs = F.softmax(edge_type_logits, dim=-1)
        
        return node_type_probs, edge_type_probs
    
