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
        config_name="train" # use train parameters as base
        )
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    log.info("Training policy...")
    
    wandb.init(
        project=cfg.policy,
        group=cfg.task.task_name,
        name=f"v1.1.3 - E_GNN Encoder",
        # track hyperparameters and run metadata
        config={
            "policy": cfg.policy,
            "dataset_type": cfg.task.dataset_type,
            "n_epochs": cfg.num_epochs,
            "seed": cfg.seed,
            "task": cfg.task.task_name,
        },
        # mode="disabled",
    )
    # wandb.watch(policy.model, log="all")
    # override Hydra parameters with WandB parameters
    cfg.policy.lr = wandb.config.lr
    cfg.policy.denoising_network.hidden_dim = wandb.config.hidden_dim
    cfg.policy.denoising_network.num_layers = wandb.config.num_layers
    cfg.policy.ckpt_path = cfg.policy.ckpt_path + f"_lr{cfg.policy.lr}_hd{wandb.config.hidden_dim}_nl{wandb.config.num_layers}"

    # instanciate policy from cfg file
    policy = hydra.utils.instantiate(cfg.policy)
    log.info(f"Training policy {policy.__class__.__name__} with seed {cfg.seed} on task {cfg.task.task_name}")
    try:
        if cfg.policy.ckpt_path is not None:
            policy.load_nets(cfg.policy.ckpt_path)
    except:
        log.error("cfg.policy.ckpt_path doesn't exist")
    

    

    

    E = cfg.num_epochs
    if cfg.eval_params != "disabled":
        E = cfg.eval_params.eval_every
    
     # evaluate every E epochs
    for i in range(cfg.num_epochs // E):
        # train policy
        policy.train(dataset=policy.dataset.shuffle(),
                    num_epochs=E,
                    model_path=cfg.policy.ckpt_path ,
                    seed=cfg.seed)
        if cfg.eval_params != "disabled":
            eval_main(cfg.eval_params)

    # final epochs and evaluation
    policy.train(dataset=policy.dataset.shuffle(),
                    num_epochs=cfg.num_epochs % E,
                    model_path=cfg.policy.ckpt_path,
                    seed=cfg.seed)
    eval_main(cfg.eval_params)

    wandb.finish()

if __name__ == "__main__":
    wandb.login()


    sweep_configuration = {
        "method": "random",
        "metric": {"name": "loss", "goal": "minimize"},
        "parameters": {
            "num_epochs": 2,
            "task": "lift_graph",
            "policy": "graph_diffusion_policy",
            "lr": {"values": [0.0005, 0.0001, 0.00001, 0.000005]},
            "hidden_dim": {"values": [16, 32, 64, 128]},
            "num_layers": {"values": [1, 2, 3, 4, 5]},
        },
    }
        # 3: Start the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="E-GNN-Encoder-Sweep")
    wandb.agent(sweep_id, function=train, count=20)


