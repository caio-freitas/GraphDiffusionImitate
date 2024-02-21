'''
Script to evaluate a specific policy
Usage: python eval.py
'''

import logging
import pathlib

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
        version_base=None,
        config_path=str(pathlib.Path(__file__).parent.joinpath('imitation','config')), 
        config_name="eval"
        )
def eval_main(cfg):
    print(OmegaConf.to_yaml(cfg))
    log.info("Running evaluation...")
    # instanciate environment runner from cfg file
    runner = hydra.utils.instantiate(cfg.env_runner)
    # instanciate policy from cfg file
    policy = hydra.utils.instantiate(cfg.policy)
    # instanciate agent from policy
    agent = hydra.utils.instantiate(cfg.agent, policy=policy, env=runner.env)

    if __name__ == "__main__":
        wandb.init(
            project=policy.__class__.__name__,
            group=cfg.task.task_name,
            name=f"eval",
            # track hyperparameters and run metadata
            config={
                "policy": cfg.policy,
                "dataset_type": cfg.task.dataset_type,
                "episodes": cfg.num_episodes,
                "task": cfg.task.task_name,
            },
            # mode="disabled",
        )

    # run policy in environment
    success_count = 0
    for i in range(cfg.num_episodes):
        runner.reset()
        rewards, info = runner.run(agent, cfg.max_steps)
        assert "success" in info, "info['success'] not returned in info from runner"
        print(f"info: {info}")
        if info["success"]:
            success_count += 1
        log.info({"episode_reward": sum(rewards), "success": info["success"]})
        wandb.log({"episode_reward": sum(rewards), "success": 1 if info["success"] else 0})
        if i >= 1:
            runner.output_video = False
    log.info(f"Success rate: {success_count/cfg.num_episodes}")
    wandb.log({"success_rate": success_count/cfg.num_episodes})

if __name__ == "__main__":
    eval_main()