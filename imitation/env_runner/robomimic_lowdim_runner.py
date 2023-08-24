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
                action_horizon=1,
                render=True,
                fps=30) -> None:
        super().__init__(output_dir)
        self.env = env
        self.action_horizon = action_horizon
        self.render = render
        self.fps = fps
        self.obs = self.env.reset()

    def reset(self) -> None:
        self.obs = self.env.reset()

    def run(self, agent: BaseAgent, n_steps: int) -> Dict:
        log.info(f"Running agent {agent.__class__.__name__} for {n_steps} steps")
        done = False
        info = {}
        rewards = []
        for i in range(n_steps):
            actions = agent.act(self.obs)
            for j in range(self.action_horizon):
                # Make sure the action is always [[...]]
                if len(actions.shape) == 1:
                    log.warning("Action shape is 1D, adding batch dimension")
                    actions = actions.reshape(1, -1)
                action = actions[j] 
                if done:
                    break
                self.obs, reward, done, info = self.env.step(action)
                rewards.append(reward)
                if self.render:
                    self.env.render()
                    time.sleep(1/self.fps)
                i += 1
        self.env.close()
        return rewards, info