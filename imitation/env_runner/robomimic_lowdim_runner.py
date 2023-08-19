import logging
import time
from typing import Dict

import gymnasium as gym

from imitation.agent.base_agent import BaseAgent
from imitation.env_runner.base_runner import BaseRunner

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
            for j in range(self.action_horizon):
                # Make sure the action is always [[...]]
                action = actions[j] 
                if done:
                    break
                self.obs, reward, done, info = self.env.step(action)
                self.env.render()
                time.sleep(1/30) # TODO parametrize
                i += 1
        self.env.close()

