from typing import Dict
import gymnasium as gym
from imitation.agent.base_agent import BaseAgent
from imitation.env_runner.base_runner import BaseRunner
from imitation.policy.base_policy import BasePolicy


class KitchenPoseRunner(BaseRunner):
    def __init__(self, output_dir) -> None:
        super().__init__(output_dir)
        self.env = gym.make(
            'FrankaKitchen-v1',
            tasks_to_complete=['microwave', 'kettle'],
            render_mode='human'
            )

        self.obs = self.env.reset()

    def reset(self) -> None:
        self.obs = self.env.reset()

    def run(self, agent: BaseAgent, n_steps: int) -> Dict:
        print(self.env.metadata["render_modes"])

        for i in range(n_steps):
            self.env.render()
            self.obs = self.env.step(agent.act(self.obs))
            print(self.obs[0]["observation"][:10])
        self.env.close()

