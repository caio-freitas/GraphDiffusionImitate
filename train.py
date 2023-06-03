"""
Usage:
Training:
python train.py --config-name=example
"""
from omegaconf import DictConfig, OmegaConf
from imitation.env_runner.kitchen_pose_runner import KitchenPoseRunner
import hydra

@hydra.main(version_base=None)
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    train()