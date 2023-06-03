from imitation.policy.base_policy import BasePolicy

class RandomPolicy(BasePolicy):
    def __init__(self, env):
        super().__init__(env)

    def predict_action(self, obs):
        return self.action_space.sample()
