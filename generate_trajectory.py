"""
Usage:
Training:
python demo_env.py
"""
import logging
import pathlib
import random
import time
from typing import Optional

import numpy as np
import torch
from imitation.env.pybullet.se2_envs.robot_se2_pickplace import SE2BotPickPlace
from imitation.utils.stochgpmp import plan_stochgpmp, plot_trajectory
from omegaconf import DictConfig
from robot_envs.pybullet.utils import random_init_static_sphere
from torch_kinematics_tree.geometrics.spatial_vector import x_rot, y_rot, z_rot
from torch_kinematics_tree.models.robot_tree import DifferentiableTree


class DifferentiableSE2(DifferentiableTree):
    def __init__(self, link_list: Optional[str] = None, device='cpu'):
        robot_file = "./assets/robot/se2_bot_description/robot/robot.urdf"
        robot_file = pathlib.Path(robot_file)
        self.model_path = robot_file.as_posix()
        self.name = "differentiable_2_link_planar"
        super().__init__(self.model_path, self.name, link_list=link_list, device=device)


log = logging.getLogger(__name__)


EPISODES = 3
MAX_STEPS = 1000




def demo():
    log.info("Running demo...")
    device = torch.device('cpu')
    tensor_args = {'device': device, 'dtype': torch.float32}


    seed = int(time.time())
    num_particles_per_goal = 10
    num_samples = 32
    num_obst = 3
    traj_len = 64
    dt = 0.05 # TODO consider this in planner
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # spawn obstacles
    obst_r = [0.1, 0.2]
    obst_range_lower = np.array([0.6, -0.2, 0])
    obst_range_upper = np.array([1., 0.2, 0])
    obstacle_spheres = np.zeros((1, num_obst, 4))
    for i in range(num_obst):
        r, pos = random_init_static_sphere(obst_r[0], obst_r[1], obst_range_lower, obst_range_upper, 0.01)
        obstacle_spheres[0, i, :3] = pos
        obstacle_spheres[0, i, 3] = r
    
    env = SE2BotPickPlace(objects_list=['cube' for i in range((obstacle_spheres.shape[1]))],
                          obj_poses=[[obstacle_spheres[0][i,:3], [0,0,0,1]] for i in range(obstacle_spheres.shape[1])])
    
    # env = SE2BotPickPlace()
    env.reset()

    env.setControlMode("position")

    # FK
    robot_fk = DifferentiableSE2()
    # start & goal
    start_q = torch.tensor(env.getJointStates()[0],**tensor_args)
    start_state = torch.cat((start_q, torch.zeros_like(start_q)))
    # use IK solution from pybullet

    # print info about the robot
    log.info("Environment info:")
    log.info(f"Robot with {env.dof} DOF, control mode: {env.control_mode}")
    log.info(f"Robot joint IDs: {env.JOINT_ID}")


    # world setup (target_pos & target_rot can be randomized)
    target_pos = np.array([0.0, 1.0, 0.0])
    target_rot = (z_rot(-torch.tensor(torch.pi)) @ y_rot(-torch.tensor(torch.pi))).to(**tensor_args)
    
    pos, vel = plan_stochgpmp(
        env,
        robot_fk,
        start_state=start_state,
        target_pos=target_pos,
        target_rot=target_rot,
        num_particles_per_goal=num_particles_per_goal,
        num_samples=num_samples,
        tensor_args=tensor_args,
        traj_len=traj_len,
        dt=dt,
        obstacle_spheres=obstacle_spheres,
        seed=seed

    )

    # Plotting
    start_q = start_state.detach().cpu().numpy()
    env.step(start_q)
    trajs = pos.detach()

    for traj in trajs:
        log.info("Restarting position")
        env.reset()
        env.step(start_q)
        time.sleep(0.2)
        traj = traj.mean(dim=0)
        for t in range(traj.shape[0] - 1):
            for i in range(10):
                env.step(traj[t])
                time.sleep(0.01)
            time.sleep(dt)

        # final position
        for i in range (100):
            env.step(traj[-1])
            time.sleep(0.01)
        time.sleep(1)
        plot_trajectory(
            robot_fk,
            start_q,
            traj,
            target_pos,
            obstacle_spheres
        )

        

if __name__ == "__main__":
    demo()