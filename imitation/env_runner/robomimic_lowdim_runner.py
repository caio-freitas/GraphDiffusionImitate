import logging
from typing import Dict

import gymnasium as gym

from imitation.agent.base_agent import BaseAgent
from imitation.env.pybullet.se2_envs.robot_se2_wrapper import \
    RobotSe2EnvWrapper
from imitation.env_runner.base_runner import BaseRunner
from imitation.policy.base_policy import BasePolicy

log = logging.getLogger(__name__)

class RobomimicEnvRunner(BaseRunner):
    def __init__(self,
                env,
                output_dir,
                action_horizon=1) -> None:
        super().__init__(output_dir)
        self.env = env
        self.action_horizon = action_horizon
        self.obs = self.env.reset()

    def reset(self) -> None:
        self.obs = self.env.reset()

    def run(self, agent: BaseAgent, n_steps: int) -> Dict:
        log.info(f"Running agent {agent.__class__.__name__} for {n_steps} steps")
        done = False
        for i in range(n_steps):
            self.env.render()
            actions = agent.act(self.obs)
            for i in range(self.action_horizon):
                # Make sure the action is always [[...]]
                action = actions[i] 
                if done:
                    break
                log.info(f"action: {action}")
                self.obs, reward, done, info = self.env.step(action)

        self.env.close()

