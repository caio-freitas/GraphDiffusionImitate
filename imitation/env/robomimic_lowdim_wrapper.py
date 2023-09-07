
import gymnasium as gym
import numpy as np

import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper


class RobomimicGymWrapper(GymWrapper):
    # override the default observation keys
    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)

class RobomimicLowdimWrapper(gym.Env):
    def __init__(self,
                 max_steps=5000,
                 task="Lift",
                 robots=["Panda"],
                 ):
        controller_config = load_controller_config(default_controller="OSC_POSE")
        robots = [*robots] # gambiarra to make it work with param list
        self.env = RobomimicGymWrapper(
            suite.make(
                task,
                robots=robots,
                use_camera_obs=False,  # do not use pixel observations
                has_offscreen_renderer=False,  # not needed since not using pixel obs
                has_renderer=True,  # make sure we can render to the screen
                reward_shaping=True,  # use dense rewards
                control_freq=20,  # control should happen fast enough so that simulation looks smooth
                horizon=max_steps,  # long horizon so we can sample high rewards
                controller_configs=controller_config,
            ),
            keys = [
                "robot0_proprio-state",
                # "robot1_proprio-state", # TODO add robot1 proprio-state to observations
                "object-state"
            ]
        )
        self.env.reset()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        

    def _robosuite_obs_to_robomimic_obs(self, obs):
        '''
        Converts robosuite Gym Wrapper observations to robomimic's ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object']
        * Won't work if parameter obs_keys is changed!
          according to https://robosuite.ai/docs/modules/environments.html
        '''
        # 7  - sin of joint angles
        robot_joint_sin = obs[0:7]
        # 7  - cos of joint angles
        robot_joint_cos = obs[7:14]
        # 7  - joint velocities
        robot_joint_vel = obs[14:21]
        eef_pose = obs[21:24]
        eef_quat = obs[24:28]
        gripper_pose = obs[28:30]
        # ignore 2 - gripper joint velocities
        objects = obs[32:]
        return [*robot_joint_cos, *robot_joint_sin, *robot_joint_vel, *eef_pose, *eef_quat, *gripper_pose, *objects]
    
    def reset(self):
        obs, _ =  self.env.reset()
        return self._robosuite_obs_to_robomimic_obs(obs)
    

    def step(self, action):
        obs, reward, done, sla, info = self.env.step(action)
        obs = self._robosuite_obs_to_robomimic_obs(obs)
        return obs, reward, done, info
    
    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

