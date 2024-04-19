import logging
import timeit
import pathlib
import numpy as np
import hydra
import torch
import wandb

from omegaconf import DictConfig, OmegaConf

try:
    from torch_robotics.robots.robot_panda import RobotPanda
except ImportError:
    from torch_robotics.robots.robot_panda import RobotPanda # for some reason, the import only works the second time

from torch_robotics.trajectory.metrics import compute_smoothness, compute_path_length, compute_variance_waypoints

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("eval", eval, replace=True)



@hydra.main(
        version_base=None,
        config_path=str(pathlib.Path(__file__).parent.joinpath('imitation','config')), 
        config_name="train"
        )
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # instanciate policy from cfg file
    policy = hydra.utils.instantiate(cfg.policy)
    log.info(f"Evaluating trajectory quality for policy {policy.__class__.__name__} with seed {cfg.seed} on task {cfg.task.task_name}")
    try:
        if cfg.policy.ckpt_path is not None:
            policy.load_nets(cfg.policy.ckpt_path)
    except:
        log.error("cfg.policy.ckpt_path doesn't exist")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    robot = RobotPanda(tensor_args={"device": device})

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)

    # Split the dataset into train and validation
    train_dataset, val_dataset = torch.utils.data.random_split(
        policy.dataset, [len(policy.dataset) - int(cfg.val_fraction * len(policy.dataset)), int(cfg.val_fraction * len(policy.dataset))]
    )


    delta_traj = []
    generated_traj = []
    # evaluate for the whole dataset
    for i in range(len(policy.dataset[:10])):
        obs_deque = policy.dataset.to_obs_deque(policy.dataset[i])
        # compare the action with the ground truth action
        groundtruth_traj = policy.dataset.get_action(policy.dataset[i])

        # generate action multiple times to get multimodality
        mm_traj = []
        times = [] # to store the time taken to generate the action
        for seed in range(50):
            # change seed
            # torch.manual_seed(seed)
            start_time = timeit.default_timer()
            action = policy.get_action(obs_deque, seed)
            end_time = timeit.default_timer()
            execution_time = end_time - start_time
            times.append(execution_time)
            mm_traj.append(action)
            # compute the difference against the ground truth
            error = action - groundtruth_traj
            delta_traj.append(error)

        times = np.array(times)
        log.info(f"Average execution time for 50 : {times.mean()}")        
        mm_traj = torch.tensor(mm_traj)
        
        # calculate Waypoint Variance: sum (along the trajectory dimension) of the pairwise L2- distance 
        # variance between waypoints at corresponding time
        waypoint_variance = compute_variance_waypoints(mm_traj, robot)
        log.info(f"Waypoint Variance: {waypoint_variance}")
        # calculate Smoothness: sum (along the trajectory dimension) of the pairwise L2- distance between
        # consecutive waypoints
        smoothness = compute_smoothness(mm_traj, robot)
        # compute average smoothness over trajectories
        smoothness = smoothness.mean()
        log.info(f"Smoothness: {smoothness}")
        
    
    # compute the mean and std of the error
    delta_traj = np.array(delta_traj)
    mean_error = np.mean(np.mean(delta_traj, axis=1), axis=0)
    std_error = np.std(np.mean(delta_traj, axis=1), axis=0)
    log.info(f"Mean error: {mean_error}")
    log.info(f"Std error: {std_error}")

    # calculate multimodality
    multimodality = np.linalg.norm(std_error)
    

        
    


if __name__ == "__main__":
    train()


