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

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("eval", eval)

os.environ["WANDB_DISABLED"] = "false"


@hydra.main(
        version_base=None,
        config_path=str(pathlib.Path(__file__).parent.joinpath('imitation','config')), 
        config_name="train_pusht_diffusion"
        )
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    log.info("Training policy...")
    # instanciate environment runner from cfg file
    runner = hydra.utils.instantiate(cfg.env_runner)
    # instanciate policy from cfg file
    policy = hydra.utils.instantiate(cfg.policy, env=runner.env)
    log.info(f"Training policy {policy.__class__.__name__} with seed {cfg.seed} on task {cfg.task.task_name}")
    try:
        if cfg.policy.ckpt_path is not None:
            policy.load_nets(cfg.policy.ckpt_path)
    except:
        log.error("cfg.policy.ckpt_path doesn't exist")
    
    wandb.init(
        project=cfg.task.task_name,
        group=policy.__class__.__name__,
        name=f"seed_{cfg.seed}",
        # track hyperparameters and run metadata
        config={
            "policy": cfg.policy,
            "n_epochs": cfg.num_epochs,
            "episodes": len(policy.dataset),
            "batch_size": policy.dataloader.batch_size,
            "env": runner.env.__class__.__name__,
        }
    )
    # train policy

    policy.train(dataset=policy.dataset,
                 num_epochs=cfg.num_epochs,
                 model_path=cfg.model_path,
                 seed=cfg.seed)

if __name__ == "__main__":
    train()


