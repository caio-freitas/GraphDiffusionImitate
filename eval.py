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

OmegaConf.register_new_resolver("eval", eval)

@hydra.main(
        version_base=None,
        config_path=str(pathlib.Path(__file__).parent.joinpath('imitation','config')), 
        config_name="eval_lift_mlp"
        )
def test(cfg):
    print(OmegaConf.to_yaml(cfg))
    log.info("Running evaluation...")
    # instanciate environment runner from cfg file
    runner = hydra.utils.instantiate(cfg.env_runner)
    # instanciate policy from cfg file
    policy = hydra.utils.instantiate(cfg.policy, env=runner.env)
    # instanciate agent from policy
    agent = hydra.utils.instantiate(cfg.agent, policy=policy)

    # run policy in environment
    success_count = 0
    for i in range(cfg.num_episodes):
        runner.reset()
        rewards, info = runner.run(agent, cfg.max_steps)
        print(f"info: {info}")
        if info["success"]:
            success_count += 1
        log.info({"episode_reward": sum(rewards), "success": info["success"]})
    log.info(f"Success rate: {success_count/cfg.num_episodes}")

if __name__ == "__main__":
    test()