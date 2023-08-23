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

from imitation.env_runner.kitchen_pose_runner import KitchenPoseRunner
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
    log.info("Running train...")
    # instanciate environment runner from cfg file
    runner = hydra.utils.instantiate(cfg.env_runner)
    # instanciate policy from cfg file
    policy = hydra.utils.instantiate(cfg.policy, env=runner.env)
    try:
        if cfg.policy.ckpt_path is not None:
            policy.load_nets(cfg.policy.ckpt_path)
    except:
        log.error("cfg.policy.ckpt_path doesn't exist")
    
    wandb.init(
        project="se2",
        name="mlp_overfit",
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
                 model_path=cfg.model_path)

if __name__ == "__main__":
    train()


