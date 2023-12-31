from abc import ABC, abstractmethod
from typing import Dict
import torch

class BasePolicy(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def get_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Given an observation, return an action.
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

    ## TODO normalizing action