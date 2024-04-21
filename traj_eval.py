import logging
import timeit
import pathlib
import numpy as np
import hydra
import torch
import wandb

from omegaconf import DictConfig, OmegaConf


from imitation.utils.metrics import compute_variance_waypoints, compute_smoothness_from_vel, compute_smoothness_from_pos

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("eval", eval, replace=True)



@hydra.main(
        version_base=None,
        config_path=str(pathlib.Path(__file__).parent.joinpath('imitation','config')), 
        config_name="traj_eval"
        )
def traj_eval(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # instanciate policy from cfg file
    policy = hydra.utils.instantiate(cfg.policy)
    log.info(f"Evaluating trajectory quality for policy {policy.__class__.__name__} with seed {cfg.seed} on task {cfg.task.task_name}")
    try:
        if cfg.policy.ckpt_path is not None:
            policy.load_nets(cfg.policy.ckpt_path)
    except:
        log.error("cfg.policy.ckpt_path doesn't exist")

    if __name__ == "__main__":
        wandb.init(
            project=policy.__class__.__name__,
            group=cfg.task.task_name,
            name=f"traj_eval",
            # track hyperparameters and run metadata
            config={
                "policy": cfg.policy,
                "dataset_type": cfg.task.dataset_type,
                "task": cfg.task.task_name,
            },
            # mode="disabled",
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)

    step_log = {} # for wandb logging

    delta_traj = []
    generated_traj = []
    # evaluate for the whole dataset
    for i in range(len(policy.dataset[:cfg.num_episodes])):
        obs_deque = policy.dataset.to_obs_deque(policy.dataset[i])
        # compare the action with the ground truth action
        groundtruth_traj = policy.dataset.get_action(policy.dataset[i])

        # generate action multiple times to get multimodality
        mm_traj = []
        times = [] # to store the time taken to generate the action
        for seed in range(cfg.num_seeds):
            # change seed
            torch.manual_seed(seed)
            start_time = timeit.default_timer()
            action = policy.get_action(obs_deque)
            end_time = timeit.default_timer()
            execution_time = end_time - start_time
            times.append(execution_time)
            mm_traj.append(action)
            # compute the difference against the ground truth
            error = action - groundtruth_traj
            delta_traj.append(error)

        times = np.array(times)
        log.info(f"Average execution time for 50 : {times.mean()}")
        step_log["execution_time"] = times.mean()
         
        mm_traj = torch.tensor(mm_traj)
        
        # calculate Waypoint Variance: sum (along the trajectory dimension) of the pairwise L2- distance 
        # variance between waypoints at corresponding time
        waypoint_variance = compute_variance_waypoints(mm_traj)
        log.info(f"Mean Waypoint Variance: {waypoint_variance/mm_traj.shape[1]}")
        step_log["waypoint_variance"] = waypoint_variance
        step_log["mean_waypoint_variance"] = waypoint_variance/mm_traj.shape[1] # mean over time (trajectory) dimension
        # calculate Smoothness: sum (along the trajectory dimension) of the pairwise L2- distance between
        # consecutive waypoints
        if hasattr(cfg.task, 'control_mode'):
            if cfg.task.control_mode == "JOINT_POSITION":
                smoothness = compute_smoothness_from_pos(mm_traj)
            elif cfg.task.control_mode == "JOINT_VELOCITY":
                smoothness = compute_smoothness_from_vel(mm_traj)
        else: # lowdim task - default is velocity
            smoothness = compute_smoothness_from_vel(mm_traj)
        
        # compute average smoothness over trajectories
        smoothness = smoothness.mean()
        step_log["smoothness"] = smoothness
        log.info(f"Smoothness: {smoothness}")
        wandb.log(step_log)
        step_log = {}
    
    # compute the mean and std of the error
    delta_traj = np.array(delta_traj)
    mean_error = np.mean(np.mean(delta_traj, axis=1), axis=0)
    std_error = np.std(np.mean(delta_traj, axis=1), axis=0)
    log.info(f"Mean error: {mean_error}")
    log.info(f"Std error: {std_error}")
    wandb.log({"mean_error": mean_error, "std_error": std_error})

        
    


if __name__ == "__main__":
    traj_eval()


