from imitation.agent.base_agent import BaseAgent

class KitchenAgent(BaseAgent):
    def __init__(self, policy):
        super().__init__(policy)

    def act(self, observation):
        return self.policy.predict_action(observation)
    
    def reset(self):
        self.policy.reset()
        self.env.reset()