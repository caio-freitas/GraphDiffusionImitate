import gymnasium as gym
import numpy as np
import pathlib
import pybullet as p
import torch
import time
from typing import Optional

from torch_kinematics_tree.models.robot_tree import DifferentiableTree

from robot_envs.pybullet.utils import random_init_static_sphere


from imitation.env.pybullet.se2_envs.robot_se2_pickplace import SE2BotPickPlace

class DifferentiableSE2(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu'):
        robot_file = "./assets/robot/se2_bot_description/robot/robot.urdf"
        robot_file = pathlib.Path(robot_file)
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_2_link_planar"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)



class RobotSe2EnvWrapper(gym.Env):
    def __init__(self,
                 num_obs=2,
                 start_pose=[0,0,-1],
                 start_quat=[0,0,0,1]):
        self._generate_obstacle_spheres(num_obs)
        self.env = SE2BotPickPlace(objects_list=['cube' for i in range((self.obstacle_spheres.shape[1]))],
                          obj_poses=[[self.obstacle_spheres[0][i,:3], [0,0,0,1]] for i in range(self.obstacle_spheres.shape[1])])
        
        self.N_DOF = self.env.dof
        device = torch.device('cpu')
        tensor_args = {'device': device, 'dtype': torch.float32}

        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.N_DOF,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.N_DOF,), dtype=np.float32) # TODO what else for observation space?

        self.env.reset()

        self.env.setControlMode("position")

        # robot's forward kinematics
        self.robot_fk = DifferentiableSE2(device='cpu')

        self.state = self.env.getJointStates()[0]

        # start_pose = torch.tensor(start_pose, **tensor_args)
        # start_quat = torch.tensor(start_quat, **tensor_args)
        # start_joints = p.calculateInverseKinematics(self.env,
        #                                         self.env.JOINT_ID[-1],
        #                                         start_pose, 
        #                                         start_quat)[:self.env.dof]
        # self.start_joints = torch.tensor(start_joints, **tensor_args)
    
        # self.env.step(self.start_joints)



    def _generate_obstacle_spheres(self, num_obs=10):
            # spawn obstacles
            obst_r = [0.1, 0.2] # TODO add as parameter
            obst_range_lower = np.array([-1, -1, 0]) # TODO add as parameter
            obst_range_upper = np.array([1., 1, 0])
            self.obstacle_spheres = np.zeros((1, num_obs, 4))
            for i in range(num_obs):
                r, pos = random_init_static_sphere(obst_r[0], obst_r[1], obst_range_lower, obst_range_upper, 0.01)
                self.obstacle_spheres[0, i, :3] = pos
                self.obstacle_spheres[0, i, 3] = r

    def reset(self):
        [robot, obstacles, grasp_obj] = self.env.reset() # TODO use start_joints
        return robot

    def step(self, action):
        [robot, obstacles, grasp_obj] = self.env.step(action)
        '''
        TODO: Implement logic for
        - done signal
        - reward
        - info
        - observation
        '''
        time.sleep(0.05)
        return robot # pose only

    def render(self, mode="human"):
        pass

    def close(self):
        pass