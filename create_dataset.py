"""
Script to generate a dataset of trajectories using StochGPMP.

Usage:
python create_dataset.py
"""
import logging
import pathlib
import random
import time
from typing import Optional

import h5py
import hydra
import numpy as np
import pybullet as p
import torch
from imitation.env.pybullet.se2_envs.robot_se2_pickplace import SE2BotPickPlace
from imitation.utils.stochgpmp import StochGPMPSE2Wrapper, plot_trajectory
from omegaconf import DictConfig, OmegaConf
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


@hydra.main(
        version_base=None,
        config_path=str(pathlib.Path(__file__).parent.joinpath('imitation','config')), 
        config_name="stochgpmp_se2"
        )
def generate(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    log.info("Running trajectories from StochGPMP...")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    tensor_args = {'device': device, 'dtype': torch.float32}


    seed = int(time.time())
    num_particles_per_goal = cfg.num_particles_per_goal
    num_samples = cfg.num_samples
    num_obst = cfg.num_obst
    traj_len = cfg.traj_len
    dt = cfg.dt
    obstacle_spheres = np.array(cfg.obstacles) # Fixed obstacles

    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = SE2BotPickPlace(objects_list=['cube' for i in range((obstacle_spheres.shape[1]))],
                          obj_poses=[[obstacle_spheres[0][i,:3], [0,0,0,1]] for i in range(obstacle_spheres.shape[1])])

    env.setControlMode("position")

    # forward kinematic model
    robot_fk = DifferentiableSE2(device=device)
    

    # start state from config
    start_pose = torch.tensor(cfg.start_pose, **tensor_args)
    start_quat = torch.tensor(cfg.start_quat, **tensor_args)
    start_joints = p.calculateInverseKinematics(env.robot,
                                            env.JOINT_ID[-1],
                                            start_pose, 
                                            start_quat)[:env.dof]
    start_joints = torch.tensor(start_joints, **tensor_args)
    
    env.reset(start_joints)

    # start state from simulation 
    start_q = torch.tensor(env.getJointStates()[0],**tensor_args)
    start_state = torch.cat((start_q, torch.zeros_like(start_q)))

    # print info about the robot
    log.info("Environment info:")
    log.info(f"Robot with {env.dof} DOF, control mode: {env.control_mode}")
    log.info(f"Robot joint IDs: {env.JOINT_ID}")


    # world setup (target_pos & target_rot can be randomized)
    target_pos = np.array(cfg.target_pose)
    target_rot = (z_rot(-torch.tensor(torch.pi)) @ y_rot(-torch.tensor(torch.pi))).to(**tensor_args)
    
    planner = StochGPMPSE2Wrapper(
        env,
        robot_fk,
        start_state,
        tensor_args,
        seed,
        cfg.stochgpmp_params
    )

    log.info(planner)
    pos, vel = planner.plan_stochgpmp(
        target_pos=target_pos,
        target_rot=target_rot,
        num_particles_per_goal=num_particles_per_goal,
        num_samples=num_samples,
        traj_len=traj_len,
        dt=dt,
        obstacle_spheres=obstacle_spheres,
        opt_iters=cfg.opt_iters,
        sigma_self=cfg.sigma_self,
        sigma_coll=cfg.sigma_coll,
        sigma_goal=cfg.sigma_goal,
        sigma_goal_prior=cfg.sigma_goal_prior,
        sigma_start=cfg.sigma_start,
        sigma_gp=cfg.sigma_gp,
    )

    # Plotting
    start_q = start_state.detach().cpu().numpy()
    env.step(start_q)

    pos = pos.detach()
    vel = vel.detach()
    pos = pos.mean(dim=0) # mean over goals (same as pos[0] for the single-goal case)
    vel = vel.mean(dim=0)
    complete_traj = torch.cat((pos, vel), dim=-1)
    observations = torch.empty((complete_traj.shape[0], complete_traj.shape[1]-1, complete_traj.shape[2]))
    actions = torch.empty((complete_traj.shape[0], complete_traj.shape[1]-1, complete_traj.shape[2]))
    horizon = cfg.obs_horizon
    observations = complete_traj[:, :complete_traj.shape[1]-horizon, :]
    actions = complete_traj[:, horizon:, :]
    # save trajectories
    # structure from https://robomimic.github.io/docs/datasets/overview.html
    with h5py.File('./data/trajs.hdf5', 'w') as f:
        data = f.create_group('data')
        data.attrs['env_args'] = OmegaConf.to_yaml(cfg)
        for i in  range(len(actions)):
            observation = observations[i]
            act = actions[i]
            demo_i = data.create_group(f'demo_{i}')
            demo_i.attrs["num_samples"] = observation.shape[0]
            demo_i.attrs["obs_horizon"] = horizon
            demo_i.attrs["model_file"] = robot_fk.model_path # or robot_file

            demo_i.create_dataset('states', data=observation.detach().cpu().numpy())
            demo_i.create_dataset('actions', data=act.detach().cpu().numpy())
            demo_i.create_dataset('rewards', data=np.zeros((observation.shape[0], 1)))
            demo_i.create_dataset('dones', data=np.zeros((observation.shape[0], 1)))
            
            obs = demo_i.create_group('obs')
            obs.create_dataset('joint_values', data=observation[:,:3].detach().cpu().numpy())
            obs.create_dataset('joint_velocities', data=observation[:,3:].detach().cpu().numpy())
            obs.create_dataset('obstacle_spheres', data=obstacle_spheres)
            
        mask = data.create_group("mask")
        mask.create_dataset('train', data=np.ones((observations.shape[0], 1)))
        mask.create_dataset('val', data=np.zeros((observations.shape[0], 1)))
        mask.create_dataset('test', data=np.zeros((observations.shape[0], 1)))



        

if __name__ == "__main__":
    generate()