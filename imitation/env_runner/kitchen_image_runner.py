import logging
from typing import Dict
import gymnasium as gym
from imitation.agent.base_agent import BaseAgent
from imitation.env_runner.base_runner import BaseRunner
from imitation.policy.base_policy import BasePolicy
from imitation.env.kitchen_pose.kitchen_wrappers import KitchenImageWrapper

log = logging.getLogger(__name__)

class KitchenImageRunner(BaseRunner):
    def __init__(self, output_dir) -> None:
        super().__init__(output_dir)
        self.env = KitchenImageWrapper(
            gym.make(
                'FrankaKitchen-v1',
                tasks_to_complete=['microwave', 'kettle'],
                render_mode='rgb_array',
                width=256,
                height=360
                )
            )
        self.obs = self.env.reset()

    def reset(self) -> None:
        self.obs = self.env.reset()

    def run(self, agent: BaseAgent, n_steps: int) -> Dict:
        log.info(f"Running agent {agent.__class__.__name__} for {n_steps} steps")
        for i in range(n_steps):
            self.obs = self.env.render()
            self.env.step(agent.act(self.obs))
        self.env.close()

