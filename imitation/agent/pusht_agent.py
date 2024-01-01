from imitation.agent.base_agent import BaseAgent

class PushTAgent(BaseAgent):
    def __init__(self, policy):
        super().__init__(policy)

    def get_action(self, observation):
        return self.policy.get_action(observation)
    
    def reset(self):
        self.policy.reset()
        self.env.reset()