from typing import Dict
import gymnasium as gym
from imitation.env_runner.base_runner import BaseRunner
from imitation.policy.base_policy import BasePolicy
from imitation.policy.random_policy import RandomPolicy


class KitchenPoseRunner(BaseRunner):
    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.env = gym.make(
            'FrankaKitchen-v1',
            tasks_to_complete=['microwave', 'kettle'],
            render_mode='human'
            )

        self.obs = self.env.reset()

    def reset(self) -> None:
        self.obs = self.env.reset()

    def run(self, policy: BasePolicy, n_steps: int) -> Dict:
        print(self.env.metadata["render_modes"])

        for i in range(n_steps):
            self.env.render()
            obs = self.env.step(policy.predict_action(self.obs))
            print(obs[0]["observation"][:10])
        self.env.close()

