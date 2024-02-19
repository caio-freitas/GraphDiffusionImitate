'''
Diffusion Policy for imitation learning with graphs
'''
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import wandb
import torch_geometric
from imitation.utils.graph_diffusion import NodeMasker


class StochAutoregressiveGraphDiffusionPolicy(nn.Module):
    def __init__(self,
                 dataset,
                 node_feature_dim,
                 num_edge_types,
                 denoising_network,
                 lr=1e-4,
                 ckpt_path=None,
                 device = None,
                 mode = 'joint-space'):
        super(StochAutoregressiveGraphDiffusionPolicy, self).__init__()
        if device == None:
           self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.dataset = dataset
        self.node_feature_dim = node_feature_dim
        self.num_edge_types = num_edge_types
        self.model = denoising_network
        # no need for diffusion ordering network
        self.node_feature_loss = nn.MSELoss()
        self.masker = NodeMasker(dataset)
        self.global_epoch = 0

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=50, factor=0.5, verbose=True, min_lr=lr/100)
        self.mode = mode

        if ckpt_path is not None:
            self.load_nets(ckpt_path)

    def load_nets(self, ckpt_path):
        '''
        Load networks from checkpoint
        '''
        print(f"Loading networks from checkpoint: {ckpt_path}")
        try:
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        except Exception as e:
            print(f"Could not load networks from checkpoint: {e}\n")


    def save_nets(self, ckpt_path):
        '''
        Save networks to checkpoint
        '''
        torch.save(self.model.state_dict(), ckpt_path)

    def generate_diffusion_trajectory(self, graph):
        '''
        Generate a diffusion trajectory from graph
        - Diffuse nodes representing object first
        - Then diffuse nodes representing robot joints, from end-effector to base
        '''
        diffusion_trajectory = []
        '''
            Initially permutation variant, since node ordering is known (robot nodes first, then object node(s))
        '''
        node_order = torch.arange(graph.x.shape[0]-1, -1, -1)
        diffusion_trajectory.append(graph)
        masked_data = graph.clone()
        for t in range(graph.x.shape[0]):
            node = node_order[t]
            masked_data = masked_data.clone().to(self.device)
            
            masked_data = self.masker.mask_node(masked_data, node)
            diffusion_trajectory.append(masked_data)
            # don't remove last node
            if t < len(node_order)-1:
                masked_data = self.masker.remove_node(masked_data, node)
                node_order = [n-1 if n > node else n for n in node_order] # update node order to account for removed node

        return diffusion_trajectory

    def preprocess(self, graph):
        '''
        Preprocesses graph to be used by the denoising network.
        '''
        graph = graph.clone()
        graph = self.masker.fully_connect(graph)
        graph.x = graph.x.float()
        graph.edge_attr = graph.edge_attr.long()
        return graph

    def node_decay_ordering(self, graph):
        '''
        Returns node decay ordering
        '''
        return torch.arange(graph.x.shape[0]-1, -1, -1)

    def nll_loss(self, G_0, node_features, node):
        n_i = G_0.x.shape[0]
        # get likelihood of joint values
        log_likelihood = self.get_distribution_likelihood(node_features, G_0.x[node,:,:])
        loss = - log_likelihood
        return loss

    
    def get_distribution_likelihood(self, dist_params, joint_values):
            '''
            Returns the likelihood of joint values given a Mixture of Gaussians (MoG) distribution.
            Args:
                dist_params: A tensor containing parameters for the MoG distribution.
                This should be in the format [means, variances, mixing_weights],
                where means and variances are shaped [pred_horizon, node_feature_dim, num_mixtures],
                and mixing_weights are shaped [num_mixtures].
                joint_values: A tensor containing joint values shaped [pred_horizon, node_feature_dim].
            '''
            # Extract parameters
            means = dist_params[0].to(self.device).double()
            variances = dist_params[1].to(self.device).double()
            mixing_weights = dist_params[2].to(self.device).double()

            # Reshape mixing_weights for broadcasting
            mixing_weights = mixing_weights.unsqueeze(0).unsqueeze(0)  # Add two dimensions to the front

            # Calculate squared differences and normalize by variances
            repeated_joint_values = joint_values.unsqueeze(2).repeat(1, 1, mixing_weights.shape[2])
            squared_diffs = (repeated_joint_values - means) ** 2 / variances

            # Calculate exponential terms and multiply with mixing weights
            exp_terms = torch.exp(-0.5 * squared_diffs) * mixing_weights

            # Calculate likelihoods and sum over mixtures
            likelihoods = exp_terms / torch.sqrt(2 * np.pi * variances)
            likelihood_sum = torch.sum(likelihoods, dim=2)  # sum over mixtures

            # Calculate and return log-likelihood
            return torch.sum(torch.log(likelihood_sum))


    def sample_from_distribution(self, dist_params):
        '''
        Samples joint values from a Mixture of Gaussians (MoG) distribution.
        Args:
            dist_params: A tensor containing parameters for the MoG distribution.
            This should be in the format [means, variances, mixing_weights],
            where means and variances are shaped [pred_horizon, node_feature_dim, num_mixtures],
            and mixing_weights are shaped [num_mixtures].

        Returns:
            joint_values: A tensor containing sampled joint values shaped
            [pred_horizon, num_features].
        '''

        # Extract parameters
        means = dist_params[0]
        variances = dist_params[1]
        mixing_weights = dist_params[2]

        # Get the index of the maximum mixing weight
        max_mixture_idx = torch.argmax(mixing_weights)

        # Sample from Gaussian using the maximum mixing weight
        pred_horizon = means.shape[0]
        num_features = means.shape[1]
        joint_values = torch.zeros(pred_horizon, num_features) # single joint/node
        for i in range(pred_horizon):
            for j in range(num_features):
                # Sample from Gaussian using the maximum mixing weight
                joint_values[i, j] = torch.normal(means[i, j, max_mixture_idx], torch.sqrt(variances[i, j, max_mixture_idx]))

        return joint_values

    def train(self, dataset, num_epochs=100, model_path=None, seed=0):
        '''
        Train noise prediction model
        '''
        try:
            self.load_nets(model_path)
        except:
            pass
        self.optimizer.zero_grad()
        self.model.train()
        batch_size = 5

        with tqdm(range(num_epochs), desc='Epoch', leave=False) as tepoch:
            for epoch in tepoch:
                batch_i = 0
                # batch loop
                with tqdm(dataset, desc='Batch', leave=False) as tepoch:
                    for nbatch in tepoch:
                        # preprocess graph
                        graph = self.preprocess(nbatch)
                        graph = self.masker.idxify(graph)
                        diffusion_trajectory = self.generate_diffusion_trajectory(graph)  
                        # predictions & loss
                        G_0 = diffusion_trajectory[0].to(self.device)
                        node_order = self.node_decay_ordering(G_0)
                        acc_loss = 0
                        
                        # loop over nodes
                        for t in range(len(node_order)):
                            G_pred = diffusion_trajectory[t+1].clone().to(self.device)

                            # predict node and edge type distributions
                            logits = self.model(G_pred.x.float(), G_pred.edge_index, G_pred.edge_attr.float(), cond=G_0.y.float())
                            
                            # joint_values = node_features[0] # first node feature is joint values
                            # mse loss for node features
                            # loss = self.node_feature_loss(joint_values, G_0.x[node_order[t],:,:].float())
                            loss = self.nll_loss(G_0, logits, node_order[t])

                            # TODO add loss for absolute positions, to make the model physics-informed

                            wandb.log({"epoch": self.global_epoch, "nll_loss": loss.item()})

                            acc_loss += loss.item()
                            # backprop (accumulated gradients)
                            loss.backward()
                        batch_i += 1
                        # update weights
                        if batch_i % batch_size == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            self.scheduler.step(acc_loss)
                            self.save_nets(model_path)
                            wandb.log({"batch_loss": acc_loss, "learning_rate": self.optimizer.param_groups[0]['lr']})
                self.global_epoch += 1

    def get_joint_values(self, x):
        '''
        Return joint value commands from node feature vector
        Depends on operation mode:
        - joint-space: return first element
        - task-joint-space: return first element
        - end-effector: raise NotImplementedError
        '''
        if self.mode == 'joint-space' or self.mode == 'task-joint-space':
            return x[:8,:,0].T # all nodes, all timesteps, first value
        elif self.mode == 'end-effector':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _lookup_edge_attr(self, edge_index, edge_attr, action_edge_index):
        '''
        Lookup edge attributes from obs to action
        '''
        # edge_attr = edge_attr.float()
        action_edge_attr = torch.zeros(action_edge_index.shape[1])
        for i in range(action_edge_index.shape[1]):
            # find edge in obs
            action_edge_attr[i] = edge_attr[torch.logical_and(edge_index[0] == action_edge_index[0, i], edge_index[1] == action_edge_index[1, i])]
        return action_edge_attr
    
    def get_action(self, obs):
        '''
        Get action from observation
        obs: deque of observations from lowdim runner
        '''
        # append x, edge_index, edge_attr from all graphs in obs to single graph
        assert len(obs) == self.dataset.obs_horizon
        # turn deque into single graph with extra node-feature dimension
        node_features = []
        # edge_indes and edge_attr don't change
        obs[0] = self.preprocess(obs[0])
        obs[0] = self.masker.idxify(obs[0])
        edge_index = obs[0].edge_index
        edge_attr = obs[0].edge_attr
        for i in range(len(obs)):
            node_features.append(obs[i].x.unsqueeze(1))
        node_features = torch.cat(node_features, dim=1)

       
        self.model.eval()

        # graph action representation: x, edge_index, edge_attr
        action = self.masker.create_empty_graph(1) # one masked node

        for x_i in range(obs[0].x.shape[0]): # number of nodes in action graph
            action = self.preprocess(action)
            # predict node attributes for last node in action
            logits = self.model(action.x.float(), action.edge_index, action.edge_attr, cond=node_features)
            action.x[-1]  = self.sample_from_distribution(logits)
            # map edge attributes from obs to action
            action.edge_attr = self._lookup_edge_attr(edge_index, edge_attr, action.edge_index)
            if x_i == obs[0].x.shape[0]-1:
                break
            action = self.masker.add_masked_node(action)
            
            
            
        joint_values_t = self.get_joint_values(action.x.detach().cpu().numpy())

        return joint_values_t