import torch
from torch_geometric.utils import to_dense_adj

def random_node_decay_ordering(datapoint):
    # create random list of nodes
    return torch.randperm(datapoint.x.shape[0]).tolist()

class NodeMasking:
    def __init__(self, dataset):
        self.dataset = dataset
        self.NODE_MASK = dataset.x.unique().shape[0]
        self.EDGE_MASK = 0
        self.EMPTY_EDGE_MASK = -1


    def is_masked(self, datapoint, node=None):
        '''
        returns if node is masked or not, or array of masked nodes if node == None
        '''
        if node is None:
            return datapoint.x == self.NODE_MASK
        return datapoint.x[node] == self.NODE_MASK


    def mask_node(self, datapoint, selected_node):
        '''
        Masking node mechanism
        1. Masked node (x = -1)
        2. Connected to all other nodes in graph by masked edges (edge_attr = -1)
        
        datapoint.x: node feature matrix
        datapoint.edge_index: edge index matrix
        datapoint.edge_attr: edge attribute matrix
        datapoint.y: target value
        '''
        # mask node
        datapoint = datapoint.clone()
        datapoint.x[selected_node] = self.NODE_MASK
        
        # TODO fix this, should be connected to all other nodes
        # mask edges
        datapoint.edge_attr[datapoint.edge_index[0] == selected_node] = self.EDGE_MASK
        datapoint.edge_attr[datapoint.edge_index[1] == selected_node] = self.EDGE_MASK
        return datapoint
    
    def fully_connect(self, graph):
        '''
        Fully connect graph with edge attribute value
        '''
        adjacency_matrix = to_dense_adj(graph.edge_index)[0]
        adjacency_matrix[adjacency_matrix == 0] = 1

        fully_connected = graph.clone()
        fully_connected.edge_attr = torch.ones(fully_connected.x.shape[0]**2) * self.EMPTY_EDGE_MASK
        
        fully_connected.edge_attr = fully_connected.edge_attr.long()

        # restore values of original edges
        for edge_attr, edge_index in zip(graph.edge_attr, graph.edge_index.T):
            fully_connected.edge_attr[edge_index[0] * fully_connected.x.shape[0] + edge_index[1]] = edge_attr

        fully_connected.edge_index = torch.nonzero(adjacency_matrix).T
        return fully_connected