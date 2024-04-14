
import gymnasium as gym
import numpy as np

import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper

from diffusion_policy.model.common.rotation_transformer import RotationTransformer


import logging
from tqdm import tqdm

log = logging.getLogger(__name__)

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
                    log.info("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)

class RobomimicLowdimWrapper(gym.Env):
    def __init__(self,
                 max_steps=5000,
                 task="Lift",
                 has_renderer=True,
                 robots=["Panda"],
                 output_video=False,
                 ):
        controller_config = load_controller_config(default_controller="JOINT_VELOCITY") 
        self.robots = [*robots] # gambiarra to make it work with robots list
        keys = [ "robot0_proprio-state", 
                *[f"robot{i}_proprio-state" for i in range(1, len(self.robots))],
                "object-state"]
        self.env = RobomimicGymWrapper(
            suite.make(
                task,
                robots=self.robots,
                use_camera_obs=output_video, # use when recording video
                has_offscreen_renderer=output_video, 
                has_renderer=has_renderer,  # make sure we can render to the screen
                reward_shaping=True,  # use dense rewards
                control_freq=30,  # control should happen fast enough so that simulation looks smooth
                horizon=max_steps,  # long horizon so we can sample high rewards
                controller_configs=controller_config,
                # setup camera resolution
                camera_heights=96,
                camera_widths=96,
            ),
            keys = keys
        )
        self.env.reset()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.rotation_transformer = RotationTransformer(
            from_rep="quaternion",
            to_rep="rotation_6d"
        )
        # self._test_routine() # for testing purposes
        

    def _robosuite_obs_to_robomimic_obs(self, obs):
        '''
        Converts robosuite Gym Wrapper observations to robomimic's ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object']
        * Won't work if parameter obs_keys is changed!
          according to https://robosuite.ai/docs/modules/environments.html
        '''
        final_obs = []
        for i in range(len(self.robots)):
            j = i*39
            # 7  - sin of joint angles
            # robot_joint_pos = obs[j:j + 7]
            # 7  - sin of joint angles
            # robot_joint_sin = obs[j + 7:j + 14]
            # 7  - cos of joint angles
            # robot_joint_cos = obs[j + 14:j + 21]
            # 7  - joint velocities
            robot_joint_vel = obs[j + 21:j + 28]
            eef_pose = obs[j + 28:j + 31]
            eef_quat = obs[j + 31:j + 35]
            eef_6d = self.rotation_transformer.forward(eef_quat)
            gripper_pose = obs[j + 35:j + 37]
            # Skip 2  - gripper joint velocities
            robot_i = [*robot_joint_vel, *eef_pose, *eef_6d, *gripper_pose]
            final_obs = [*final_obs, *robot_i]
        
        objects = obs[39*len(self.robots):]
        return [*final_obs, *objects]
        

    def reset(self):
        obs, _ =  self.env.reset()
        return self._robosuite_obs_to_robomimic_obs(obs)
    

    def step(self, action):
        final_action = []
        for i in range(len(self.robots)):
            '''
            Robosuite's action space is composed of 7 joint velocities and 1 gripper velocity, while 
            in the robomimic datasets, it's composed of 7 joint velocities and 2 gripper velocities (for each "finger").
            '''
            j = i*9
            robot_joint_vel = action[j:j + 7]
            robot_gripper_vel = action[j + 8] # use only last action for gripper
            final_action = [*final_action, *robot_joint_vel, robot_gripper_vel]
        obs, reward, done, _, info = self.env.step(final_action)
        if reward == 1:
            done = True
            info = {"success": True}
        else:
            info = {"success": False}
        obs = self._robosuite_obs_to_robomimic_obs(obs)
        return obs, reward, done, info
    
    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def _test_routine(self):
        '''
        Test routine to check if the environment is working properly
        and exemplify degrees of freedom of action space
        '''
        self.reset()
        action = np.zeros(self.action_space.shape)
        log.info(f"Testing action space {self.action_space}, {self.action_space.shape[0]}")
        for j in range(self.action_space.shape[0]):
            log.info(f"Testing DOF {j}")
            for t in tqdm(range(100)):
                action[j] = 2*np.sin(3.14*t/50)
                self.step(action)   
                self.render()                

