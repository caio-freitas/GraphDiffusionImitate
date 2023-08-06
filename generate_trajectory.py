"""
Usage:
Training:
python demo_env.py
"""
import time
import torch
import logging
import random
import pybullet as p
import numpy as np
import time
from typing import Optional
import matplotlib.pyplot as plt
import pathlib

from stoch_gpmp.planner import StochGPMP
from stoch_gpmp.costs.cost_functions import CostCollision, CostComposite, CostGP, CostGoal, CostGoalPrior

from torch_kinematics_tree.models.robot_tree import DifferentiableTree
from torch_planning_objectives.fields.distance_fields import EESE3DistanceField, LinkDistanceField, FloorDistanceField, LinkSelfDistanceField
from torch_kinematics_tree.geometrics.frame import Frame
from torch_kinematics_tree.geometrics.skeleton import get_skeleton_from_model
from torch_kinematics_tree.geometrics.spatial_vector import (
    z_rot,
    y_rot,
    x_rot,
)
from robot_envs.pybullet.utils import random_init_static_sphere



from tqdm.auto import tqdm
from omegaconf import DictConfig
from imitation.env.pybullet.se2_envs.robot_se2_pickplace import SE2BotPickPlace


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


def plot(robot_fk,
         start_q,
         traj,
         target_pos,
         obstacle_spheres):
    plt.figure()
    ax = plt.axes(projection='3d')
    skeleton = get_skeleton_from_model(robot_fk, start_q, robot_fk.get_link_names()) # visualize IK solution
    skeleton.draw_skeleton(color='r', ax=ax)
    for t in range(traj.shape[0] - 1):
        if t % 4 == 0:
            skeleton = get_skeleton_from_model(robot_fk, traj[t], robot_fk.get_link_names())
            skeleton.draw_skeleton(color='b', ax=ax)
        
        skeleton = get_skeleton_from_model(robot_fk, traj[-1], robot_fk.get_link_names())
        skeleton.draw_skeleton(color='g', ax=ax)
        ax.plot(target_pos[0], target_pos[1], target_pos[2], 'r*', markersize=7)
        ax.scatter(obstacle_spheres[0, :, 0], obstacle_spheres[0, :, 1], obstacle_spheres[0, :, 2], s=obstacle_spheres[0, :, 3]*2000, color='r')
        plt.show()

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
    # robot_fk.print_link_names()
    n_dof = env.dof
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
    target_frame = Frame(rot=target_rot, trans=torch.from_numpy(target_pos).to(**tensor_args), device=device)
    target_quat = target_frame.get_quaternion().squeeze().cpu().numpy()  # [x, y, z, w]
    target_H = target_frame.get_transform_matrix()  # set translation and orientation of target here


    q_goal = p.calculateInverseKinematics(env.robot,
                                          env.JOINT_ID[-1],
                                          target_pos, 
                                          target_quat)[:n_dof]
    
    q_goal = torch.tensor(q_goal, **tensor_args)
    multi_goal_states = torch.cat([q_goal, torch.zeros_like(q_goal)]).unsqueeze(0)  # put IK solution

    # Cost functions
    robot_self_link = LinkSelfDistanceField(margin=0.03)
    robot_collision_link = LinkDistanceField()
    robot_goal = EESE3DistanceField(target_H)

    # Factored Cost params
    prior_sigmas = dict(
        sigma_start=0.0001,
        sigma_gp=0.0007,
    )
    sigma_self = 0.0001
    sigma_coll = 10
    sigma_goal = 0.00007
    sigma_goal_prior = 0.0001
    # Construct cost function
    cost_prior = CostGP(
        n_dof, traj_len, start_state, dt,
        prior_sigmas, tensor_args
    )
    cost_self = CostCollision(n_dof, traj_len, field=robot_self_link, sigma_coll=sigma_self)
    cost_coll = CostCollision(n_dof, traj_len, field=robot_collision_link, sigma_coll=sigma_coll)
    cost_goal = CostGoal(n_dof, traj_len, field=robot_goal, sigma_goal=sigma_goal)
    cost_goal_prior = CostGoalPrior(n_dof, traj_len, multi_goal_states=multi_goal_states, 
                                    num_particles_per_goal=num_particles_per_goal, 
                                    num_samples=num_samples, 
                                    sigma_goal_prior=sigma_goal_prior,
                                    tensor_args=tensor_args)
    cost_func_list = [cost_prior, cost_goal_prior, cost_self, cost_coll, cost_goal]
    cost_composite = CostComposite(n_dof, traj_len, cost_func_list, FK=robot_fk.compute_forward_kinematics_all_links)
    ## Planner - 2D point particle dynamics
    stochgpmp_params = dict(
        num_particles_per_goal=num_particles_per_goal,
        num_samples=num_samples,
        traj_len=traj_len,
        dt=dt,
        n_dof=n_dof,
        opt_iters=1, # Keep this 1 for visualization
        temperature=1.,
        start_state=start_state,
        multi_goal_states=multi_goal_states,
        cost=cost_composite,
        step_size=0.2,
        sigma_start_init=0.0001,
        sigma_goal_init=0.1,
        sigma_gp_init=0.1,
        sigma_start_sample=0.0001,
        sigma_goal_sample=0.07,
        sigma_gp_sample=0.02,
        seed=seed,
        tensor_args=tensor_args,
    )
    planner = StochGPMP(**stochgpmp_params)
    obstacle_spheres = torch.from_numpy(obstacle_spheres).to(**tensor_args)

    obs = {
        'obstacle_spheres': obstacle_spheres
    }

    #---------------------------------------------------------------------------
    # Optimize
    opt_iters =  400 # 400

    with tqdm(range(opt_iters + 1), desc='Optimization Step', leave=False, ) as tstep:
        for i in tstep:
            time_start = time.time()
            planner.optimize(**obs)
            print(f'Time(s) per iter: {time.time() - time_start} sec')
            pos, vel = planner.get_recent_samples()

    # Plotting
    start_q = start_state.detach().cpu().numpy()
    env.step(start_q)
    trajs = pos.detach()
    obstacle_spheres = obstacle_spheres.detach().cpu().numpy()

    print(trajs.shape)
    for traj in trajs:
        print("Restarting position")
        # for i in range(100):
        env.reset()
        env.step(start_q)
        time.sleep(0.2)
        traj = traj.mean(dim=0)
        print(traj.shape)
        for t in range(traj.shape[0] - 1):
            for i in range(10):
                env.step(traj[t])
                time.sleep(0.01)
            time.sleep(dt)

        for i in range (100):
            env.step(traj[-1])
            time.sleep(0.01)
        time.sleep(1)
        plot(
            robot_fk,
            start_q,
            traj,
            target_pos,
            obstacle_spheres
        )

        

if __name__ == "__main__":
    demo()