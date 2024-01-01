import collections
import logging
import numpy as np
from typing import Dict
import time

from diffusion_policy.env.pusht.pusht_env import PushTEnv


from imitation.agent.base_agent import BaseAgent
from imitation.env_runner.base_runner import BaseRunner


log = logging.getLogger(__name__)

class PushtDiffusionRunner(BaseRunner):
    def __init__(self,
                 env: PushTEnv,
                 output_dir: str,
                 obs_dim: int,
                 action_dim: int,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 max_steps: int,
                 ) -> None:
        super().__init__(output_dir)

        # parameters    
        self.obs_dim        = obs_dim
        self.action_dim     = action_dim
        self.pred_horizon   = pred_horizon
        self.obs_horizon    = obs_horizon
        self.action_horizon = action_horizon
        self.max_steps      = max_steps
        self.env = env
        self.obs = self.env.reset()

        # keep a queue of last steps of observations (obs_horizon)
        self.obs_deque = collections.deque(
            [self.obs] * self.obs_horizon, maxlen=self.obs_horizon)
        

    def reset(self) -> None:
        self.obs = self.env.reset()

    def run(self, agent: BaseAgent, n_steps: int) -> Dict:
        log.info(f"Running agent {agent.__class__.__name__} for {n_steps} steps")
        agent.policy.load_nets(agent.policy.ckpt_path)
        log.info(f"Model architecture: {agent.policy}")
        done = False
        step_idx = 0
        rewards = []
        info = {}
        for i in range(n_steps):
            B = 1
            # stack the last obs_horizon number of observations
            obs_seq = np.stack(self.obs_deque)
            # normalize observation

            # run policy and get future actions 
            action_pred = agent.get_action(obs_seq)

            # only take action_horizon number of actions
            start = self.obs_horizon - 1
            end = start + self.action_horizon
            action = action_pred[start:end,:]
            # execute action_horizon number of steps
            # without replanning
            for j in range(len(action)):
                # stepping env
                obs, reward, done, info = self.env.step(action[j])
                rewards.append(reward)

                # save observations
                self.obs_deque.append(obs)
                # and reward/vis
                # rewards.append(reward)
                # imgs.append(env.render(mode='rgb_array'))
                self.env.render(mode='human')
                # update progress bar
                step_idx += 1
                # pbar.update(1)
                # pbar.set_postfix(reward=reward)
                if step_idx > self.max_steps:
                    done = True
                if done:
                    break
                i += 1
            time.sleep(0.01)
            if done:
                break
        self.env.close()
        return rewards, info
