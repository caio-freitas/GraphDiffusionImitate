import numpy as np
from imitation.policy.base_policy import BasePolicy

class RandomPolicy(BasePolicy):
    def __init__(self, action_dim, pred_horizon, lr):
        super().__init__()
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.ckpt_path = None
    def get_action(self, obs):
        # return a vector of [action_dim, pred_horizon]
        return np.random.uniform(-1, 1, size=(self.pred_horizon, self.action_dim))

    def load_nets(self, ckpt_path):
        pass
