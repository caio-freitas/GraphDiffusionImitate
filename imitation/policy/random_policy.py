from imitation.policy.base_policy import BasePolicy

class RandomPolicy(BasePolicy):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def get_action(self, obs):
        return self.env.action_space.sample()
