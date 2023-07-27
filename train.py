"""
Usage:
Training:
python train.py --config-name=example
"""
import logging
import pathlib

import hydra
from imitation.env_runner.kitchen_pose_runner import KitchenPoseRunner
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

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
    # create dataset
    policy.train(dataset=policy.dataset,
                 num_epochs=cfg.num_epochs,
                 model_path=cfg.model_path)

if __name__ == "__main__":
    train()


