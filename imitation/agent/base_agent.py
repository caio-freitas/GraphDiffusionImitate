from abc import ABC, abstractmethod
from imitation.policy.base_policy import BasePolicy
from gym.spaces import Space
from typing import List

class BaseAgent(ABC):
    '''
    Class to encapsulate the agent's interaction with the environment.
    '''
    def __init__(self, policy: BasePolicy) -> None:
        self.policy = policy
    
    @abstractmethod
    def get_action(self, observation) -> Space:
        '''
        Action is a gym.spaces.Space object.
        '''
        return self.policy.get_action(observation)
    

    @abstractmethod
    def reset(self) -> None:
        '''
        Resets the agent's state.
        '''
        self.policy.reset()

