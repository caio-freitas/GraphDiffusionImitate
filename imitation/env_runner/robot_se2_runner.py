import logging
from typing import Dict
import gymnasium as gym
from imitation.agent.base_agent import BaseAgent
from imitation.env_runner.base_runner import BaseRunner
from imitation.policy.base_policy import BasePolicy
from imitation.env.pybullet.se2_envs.robot_se2_wrapper import RobotSe2EnvWrapper

log = logging.getLogger(__name__)

class RobotSe2EnvRunner(BaseRunner):
    def __init__(self, output_dir) -> None:
        super().__init__(output_dir)
        self.env = RobotSe2EnvWrapper(
            num_obs=2
        )

        self.obs = self.env.reset()

    def reset(self) -> None:
        self.obs = self.env.reset()

    def run(self, agent: BaseAgent, n_steps: int) -> Dict:
        log.info(f"Running agent {agent.__class__.__name__} for {n_steps} steps")
        for i in range(n_steps):
            self.env.render()
            self.obs = self.env.step(agent.act(self.obs))
        self.env.close()

