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
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from imitation.env.pybullet.se2_envs.robot_se2_pickplace import SE2BotPickPlace
from imitation.utils.stochgpmp import StochGPMPSE2Wrapper, plot_trajectory
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



@hydra.main(
        version_base=None,
        config_path=str(pathlib.Path(__file__).parent.joinpath('imitation','config')), 
        config_name="stochgpmp_se2"
        )
def generate(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    log.info("Running trajectories from StochGPMP...")

    device = torch.device('cpu')
    tensor_args = {'device': device, 'dtype': torch.float32}


    seed = int(time.time())
    num_particles_per_goal = cfg.num_particles_per_goal
    num_samples = cfg.num_samples
    num_obst = cfg.num_obst
    traj_len = cfg.traj_len
    dt = cfg.dt

    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # spawn obstacles
    obst_r = [0.05, 0.1] # TODO add to config
    obst_range_lower = np.array([-0.5 , -0.5, 0])
    obst_range_upper = np.array([-0.5, 0.5, 0])
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
    # start state from simulation 
    start_q = torch.tensor(env.getJointStates()[0],**tensor_args)
    start_state = torch.cat((start_q, torch.zeros_like(start_q)))

    # start state from config
    # start_state = torch.tensor(cfg.start_state, **tensor_args)
    # env.step(start_state)

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
        opt_iters=cfg.opt_iters
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
    generate()