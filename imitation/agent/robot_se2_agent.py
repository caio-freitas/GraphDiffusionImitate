from imitation.agent.base_agent import BaseAgent

class Se2Agent(BaseAgent):
    def __init__(self, policy):
        super().__init__(policy)
        self.env = policy.env
    def act(self, observation):
        # TODO should act on environment
        return self.policy.get_action(observation)
    
    def reset(self):
        self.policy.reset()
        self.env.reset()