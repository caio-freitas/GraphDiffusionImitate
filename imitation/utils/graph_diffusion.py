import torch
from torch_geometric.utils import to_dense_adj


class NodeMasker:
    def __init__(self, dataset):
        self.dataset = dataset
        self.node_feature_dim = dataset[0].x.shape[1]
        assert dataset.edge_attr.shape[1] == 1, f"Edge feature dim is not 1, got {dataset.edge_attr.shape[1]}"  # edge_feature_dim is 1, since edge_attr is just edge type
        self.edge_type_to_idx = {edge_type.item(): idx for idx, edge_type in enumerate(dataset.edge_attr.unique())}
        self.NODE_MASK = 1000
        self.EMPTY_EDGE = dataset[0].edge_attr.unique().shape[0]
        self.EDGE_MASK = dataset[0].edge_attr.unique().shape[0] + 1
        # add masks to node and edge types
        self.edge_type_to_idx[self.EMPTY_EDGE] = len(self.edge_type_to_idx)
        self.edge_type_to_idx[self.EDGE_MASK] = len(self.edge_type_to_idx)

    def idxify(self, datapoint):
        '''
        Converts edge types to indices
        '''
        datapoint = datapoint.clone()
        datapoint.edge_attr = torch.tensor([self.edge_type_to_idx[edge_type.item()] for edge_type in datapoint.edge_attr])
        return datapoint
    
    def deidxify(self, datapoint):
        '''
        Converts edge indices to types
        '''
        datapoint = datapoint.clone()
        datapoint.edge_attr = torch.tensor([list(self.edge_type_to_idx.keys())[edge_idx] for edge_idx in datapoint.edge_attr])
        return datapoint

    def is_masked(self, datapoint, node=None):
        '''
        returns if node is masked or not, or array of masked nodes if node == None
        '''
        if node is None:
            return datapoint.x == self.NODE_MASK
        return datapoint.x[node] == self.NODE_MASK

    def remove_node(self, datapoint, node):
        '''
        Removes node from graph, and all edges connected to it
        '''
        assert node < datapoint.x.shape[0], "Node does not exist"
        if datapoint.x.shape[0] == 1:
            return datapoint.clone()
        datapoint = datapoint.clone()
        # remove node
        datapoint.x = torch.cat([datapoint.x[:node], datapoint.x[node+1:]])
        
        # remove edges from edge_index (remove elements containing node in tuple of edge_index) (if datapoint.edge_index[:, 0] == node or datapoint.edge_index[:, 1] == node)
        if datapoint.edge_index.shape[1] > 1:

            # remove edges (remove elements containing node)
            datapoint.edge_attr = torch.tensor([edge_attr for edge_attr, edge_index in zip(datapoint.edge_attr, datapoint.edge_index.T) if node not in edge_index])

            edge_index_T = torch.stack([edge_index_tuple for edge_index_tuple in datapoint.edge_index.T if node not in edge_index_tuple])
            datapoint.edge_index = edge_index_T.T
            # update indices of edge_index
            datapoint.edge_index[datapoint.edge_index > node] -= 1
        return datapoint

    def add_masked_node(self, datapoint):
        '''
        Adds a masked node to the graph
        '''
        datapoint = datapoint.clone()
        masked_node_feature = self.NODE_MASK*torch.ones(1, self.node_feature_dim)
        datapoint.x = torch.cat([datapoint.x, masked_node_feature], dim=0)
        datapoint.edge_attr = torch.cat([datapoint.edge_attr, torch.tensor([self.EDGE_MASK]).repeat(datapoint.x.shape[0]-1, 1)], dim=0)
        datapoint.edge_index = torch.cat([datapoint.edge_index, torch.tensor([(node, datapoint.x.shape[0]-1) for node in range(datapoint.x.shape[0]-1)]).T], dim=1)
        return datapoint


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
        
        # mask edges
        datapoint.edge_attr[datapoint.edge_index[0] == selected_node] = self.EDGE_MASK
        datapoint.edge_attr[datapoint.edge_index[1] == selected_node] = self.EDGE_MASK
        return datapoint
    
    def _reorder_edge_attr_and_index(self, graph):
        '''
        Reorders edge_attr and edge_index to be like on nx graph
        (0, 0), (0, 1), (0, 2), ..., (0, n), (1, 0), (1, 1), ..., (n, n)
        '''
        graph = graph.clone()
        # reorder edge_attr
        edge_attr = torch.zeros(graph.x.shape[0]**2)
        for edge_attr_value, edge_index in zip(graph.edge_attr, graph.edge_index.T):
            edge_attr[edge_index[0] * graph.x.shape[0] + edge_index[1]] = edge_attr_value
        graph.edge_attr = edge_attr.long()
        # reorder edge_index
        edge_index = torch.zeros((2, graph.x.shape[0]**2))
        for i, edge_index_tuple in enumerate(graph.edge_index.T):
            edge_index[0, i] = edge_index_tuple[0]
            edge_index[1, i] = edge_index_tuple[1]
        graph.edge_index = edge_index.long()
        return graph


    def demask_node(self, graph, selected_node, node_value, connections_types):
        '''
        Demasking node mechanism
        1. Unmasked node (x = node_value)
        2. Connected to all other nodes in graph by unmasked edges (edge_attr = connections_types)
        '''
        # demask node
        graph = graph.clone()
        graph.x[selected_node] = node_value
        # demask edge_attr
        for i, connection in enumerate(connections_types):
            if not self.is_masked(graph, node=i):
                graph.edge_attr[torch.logical_and(graph.edge_index[0] == i, graph.edge_index[1] == selected_node)] = connection
                
        # reorder edge_attr and edge_index to be like on nx graph
        graph = self._reorder_edge_attr_and_index(graph)
        return graph
    

    def fully_connect(self, graph, keep_original_edges=True):
        '''
        Fully connect graph with edge attribute value
        '''
        adjacency_matrix = to_dense_adj(graph.edge_index)[0]
        adjacency_matrix[adjacency_matrix == 0] = 1

        fully_connected = graph.clone()
        fully_connected.edge_attr = torch.ones(fully_connected.x.shape[0]**2) * self.EMPTY_EDGE
        
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
        fully_masked.x = torch.ones(n_nodes, 1, dtype=torch.int32) * self.NODE_MASK
        fully_masked = self.fully_connect(fully_masked, keep_original_edges=False)
        return fully_masked
    

    def get_unmasked_nodes(self, graph):
        '''
        Returns a list of nodes that are denoised
        '''
        unmasked_nodes = []
        for node in range(graph.x.shape[0]):
            if not self.is_masked(graph, node):
                unmasked_nodes.append(node)

        return unmasked_nodes