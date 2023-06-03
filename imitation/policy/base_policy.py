from typing import Dict
import torch

class BasePolicy:
    def __init__(self, env):
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

    ## TODO normalizing action