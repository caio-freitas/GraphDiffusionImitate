from imitation.agent.base_agent import BaseAgent
import torch

class RobomimicLowdimAgent(BaseAgent):
    def __init__(self, policy):
        super().__init__(policy)
        self.env = policy.env
    def act(self, observation, latent=None):
        if latent is None:
            return self.policy.get_action(observation)
        else:
            return self.policy.get_action(observation, latent)
    
    def reset(self):
        self.policy.reset()
        self.env.reset()