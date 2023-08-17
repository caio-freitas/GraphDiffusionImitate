
import gymnasium as gym
import numpy as np

import robosuite as suite
from robosuite.environments.manipulation.lift import Lift
from robosuite.wrappers.gym_wrapper import GymWrapper


class RobomimicLowdimWrapper(gym.Env):
    def __init__(self,
                 max_steps=5000,
                 task="Lift"
                 ):
        self.env = GymWrapper(
            suite.make(
                task,
                robots="Panda",  # use Sawyer robot
                use_camera_obs=False,  # do not use pixel observations
                has_offscreen_renderer=False,  # not needed since not using pixel obs
                has_renderer=True,  # make sure we can render to the screen
                reward_shaping=True,  # use dense rewards
                control_freq=20,  # control should happen fast enough so that simulation looks smooth
                horizon=max_steps,  # long horizon so we can sample high rewards
            )
        )
        self.env.reset()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        


    def reset(self):
        obs, _ =  self.env.reset()
        return obs[:19]
    

    def step(self, action):
        print(f"action: {action}")
        action = [*action, 0] # fix gripper action
        obs, reward, done, _, info = self.env.step(action)
        print(f"obs: {obs}")
        return obs[:19], reward, done, info
    
    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

