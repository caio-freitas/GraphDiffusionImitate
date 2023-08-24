import logging
import time
from typing import Dict

import gymnasium as gym
import collections
from imitation.agent.base_agent import BaseAgent
from imitation.env_runner.base_runner import BaseRunner

log = logging.getLogger(__name__)

class RobomimicEnvRunner(BaseRunner):
    def __init__(self,
                env,
                output_dir,
                obs_horizon=1,
                action_horizon=1) -> None:
        super().__init__(output_dir)
        self.env = env
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        # keep a queue of last obs_horizon steps of observations
        obs = self.env.reset()
        self.obs_deque = collections.deque(
            [obs] * self.obs_horizon, maxlen=self.obs_horizon)
        

    def reset(self) -> None:
        self.obs = self.env.reset()

    def run(self, agent: BaseAgent, n_steps: int) -> Dict:
        log.info(f"Running agent {agent.__class__.__name__} for {n_steps} steps")
        done = False
        for i in range(n_steps):
            self.env.render()
            actions = agent.act(self.obs_deque)
            for j in range(self.action_horizon):
                # Make sure the action is always [[...]]
                if len(actions.shape) == 1:
                    log.warning("Action shape is 1D, adding batch dimension")
                    actions = actions.reshape(1, -1)
                action = actions[j] 
                if done:
                    break
                obs, reward, done, info = self.env.step(action)
                self.obs_deque.append(obs)
                
                self.env.render()
                time.sleep(1/30) # TODO parametrize
                i += 1
        self.env.close()

