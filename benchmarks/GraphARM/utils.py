import torch
from torch_geometric.utils import to_dense_adj

def random_node_decay_ordering(datapoint):
    # create random list of nodes
    return torch.randperm(datapoint.x.shape[0]).tolist()

class NodeMasking:
    def __init__(self, dataset):
        self.dataset = dataset
        self.NODE_MASK = dataset.x.unique().shape[0]
        self.EDGE_MASK = dataset.edge_attr.unique().shape[0]
        self.EMPTY_EDGE_MASK = dataset.edge_attr.unique().shape[0] + 1


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
    
    def demask_node(self, graph, selected_node, node_type, connections_types):
        '''
        Demasking node mechanism
        1. Unmasked node (x = node_type)
        2. Connected to all other nodes in graph by unmasked edges (edge_attr = connections_types)
        '''
        # demask node
        graph = graph.clone()
        graph.x[selected_node] = node_type
        # demask edge_attr
        for i, connection in enumerate(connections_types):
            if self.is_masked(graph, node=i):
                graph.edge_attr[graph.edge_index[0] == selected_node] = connection
                graph.edge_attr[graph.edge_index[1] == selected_node] = connection

        return graph
    def fully_connect(self, graph, keep_original_edges=True):
        '''
        Fully connect graph with edge attribute value
        '''
        adjacency_matrix = to_dense_adj(graph.edge_index)[0]
        adjacency_matrix[adjacency_matrix == 0] = 1

        fully_connected = graph.clone()
        fully_connected.edge_attr = torch.ones(fully_connected.x.shape[0]**2) * self.EMPTY_EDGE_MASK
        
        fully_connected.edge_attr = fully_connected.edge_attr.long()

        if keep_original_edges:
            # restore values of original edges
            for edge_attr, edge_index in zip(graph.edge_attr, graph.edge_index.T):
                fully_connected.edge_attr[edge_index[0] * fully_connected.x.shape[0] + edge_index[1]] = edge_attr

        fully_connected.edge_index = torch.nonzero(adjacency_matrix).T
        return fully_connected
    
    def generate_fully_masked(self, like):
        '''
        Generates a fully masked graph like the one provided
        '''
        
        n_nodes = like.x.shape[0]

        fully_masked = like.clone()
        fully_masked.x = torch.ones(n_nodes) * self.NODE_MASK
        fully_masked = self.fully_connect(fully_masked, keep_original_edges=False)
        return fully_masked
    

    def get_denoised_nodes(self, graph):
        '''
        Returns a list of nodes that are denoised
        '''
        denoised_nodes = []
        for node in range(graph.x.shape[0]):
            if not self.is_masked(graph, node):
                denoised_nodes.append(node)

        return denoised_nodes