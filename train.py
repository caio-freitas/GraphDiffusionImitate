"""
Usage:
Training:
python train.py --config-name=example
"""
import logging
import os
import pathlib

import hydra
import wandb

from eval import eval_main
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
        version_base=None,
        config_path=str(pathlib.Path(__file__).parent.joinpath('imitation','config')), 
        config_name="train"
        )
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    log.info("Training policy...")
    # instanciate policy from cfg file
    policy = hydra.utils.instantiate(cfg.policy)
    log.info(f"Training policy {policy.__class__.__name__} with seed {cfg.seed} on task {cfg.task.task_name}")
    try:
        if cfg.policy.ckpt_path is not None:
            policy.load_nets(cfg.policy.ckpt_path)
    except:
        log.error("cfg.policy.ckpt_path doesn't exist")
    
    wandb.init(
        project=policy.__class__.__name__,
        group=cfg.task.task_name,
        name=f"v0.0.1",
        # track hyperparameters and run metadata
        config={
            "policy": cfg.policy,
            "n_epochs": cfg.num_epochs,
            "seed": cfg.seed,
            "lr": cfg.policy.lr,
            "episodes": len(policy.dataset),
            "task": cfg.task.task_name,
        },
        # mode="disabled",
    )
    # train policy

    policy.train(dataset=policy.dataset,
                 num_epochs=cfg.num_epochs,
                 model_path=cfg.policy.ckpt_path,
                 seed=cfg.seed)#
    if cfg.eval_params is not None:
        eval_main(cfg.eval_params)

if __name__ == "__main__":
    train()


