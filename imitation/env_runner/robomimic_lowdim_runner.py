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
                action_horizon,
                obs_horizon,
                render=True,
                fps=30) -> None:
        super().__init__(output_dir)
        self.env = env
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.render = render
        self.fps = fps
        
        # keep a queue of last obs_horizon steps of observations
        obs = self.env.reset()
        self.obs_deque = collections.deque(
            [obs] * self.obs_horizon, maxlen=self.obs_horizon)

    def reset(self) -> None:
        self.obs = self.env.reset()

    def run(self, agent: BaseAgent, n_steps: int) -> Dict:
        log.info(f"Running agent {agent.__class__.__name__} for {n_steps} steps")
        done = False
        info = {}
        rewards = []
        for i in range(n_steps):
            actions = agent.get_action(self.obs_deque)
            
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
                
                if self.render:
                    self.env.render()
                    time.sleep(1/self.fps)
                i += 1
        self.env.close()
        return rewards, info