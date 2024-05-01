"""
Usage:
Training:
python train.py --config-name=example
"""
import logging
import os
import pathlib

import hydra
import torch
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
        if cfg.policy.ckpt_path is not None and cfg.load_ckpt:
            policy.load_nets(cfg.policy.ckpt_path)
    except Exception as e:
        log.error(f"Error loading checkpoint: {e}")
    
    wandb.init(
        project=policy.__class__.__name__,
        group=cfg.task.task_name,
        name=f"v1.2.2",
        # track hyperparameters and run metadata
        config={
            "policy": cfg.policy,
            "dataset_type": cfg.task.dataset_type,
            "n_epochs": cfg.num_epochs,
            "seed": cfg.seed,
            "lr": cfg.policy.lr,
            "task": cfg.task.task_name,
        },
        # mode="disabled",
    )
    # wandb.watch(policy.model, log="all")

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)

    # Split the dataset into train and validation
    train_dataset, val_dataset = torch.utils.data.random_split(
        policy.dataset, [len(policy.dataset) - int(cfg.val_fraction * len(policy.dataset)), int(cfg.val_fraction * len(policy.dataset))]
    )

    E = cfg.num_epochs
    V = cfg.num_epochs
    if cfg.eval_params != "disabled":
        E = cfg.eval_params.eval_every
        V = cfg.eval_params.val_every
        assert V <= E, "Validation interval should be smaller than evaluation interval"
        assert E % V == 0, "Evaluation interval should be multiple of validation interval"
    
    try:
        policy.num_epochs = cfg.num_epochs
    except:
        log.error("Error setting total num_epochs in policy")

     # evaluate every E epochs
    max_success_rate = 0
    for i in range(1, 1 + cfg.num_epochs // V):
        # train policy
        policy.train(dataset=train_dataset,
                    num_epochs=V,
                    model_path=cfg.policy.ckpt_path,
                    seed=cfg.seed)
        log.info(f"Calculating validation loss...")
        val_loss = policy.validate(
                dataset=val_dataset,
                model_path=cfg.policy.ckpt_path,
            )
        wandb.log({"validation_loss": val_loss})
        # evaluate policy
        if i % (E/V) == 0:
            success_rate = eval_main(cfg.eval_params)
            if success_rate >= max_success_rate:
                max_success_rate = success_rate
                policy.save_nets(cfg.policy.ckpt_path + f"_best_succ={success_rate}.pt")

    wandb.finish()

if __name__ == "__main__":
    train()


