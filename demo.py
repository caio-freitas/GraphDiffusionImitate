"""
Usage:
Training:
python demo.py --multirun +experiment=demo,demo2
"""
import logging
from omegaconf import DictConfig, OmegaConf
from imitation.agent.kitchen_agent import KitchenAgent
from imitation.env_runner.kitchen_pose_runner import KitchenPoseRunner
from imitation.env_runner.kitchen_image_runner import KitchenImageRunner
from imitation.policy.random_policy import RandomPolicy
import hydra
import pathlib

log = logging.getLogger(__name__)

@hydra.main(
        version_base=None,
        config_path=str(pathlib.Path(__file__).parent.joinpath('imitation','config'))
        )
def demo(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    log.info("Running demo...")
    # runner = KitchenPoseRunner(cfg.experiment.output_dir)
    runner = KitchenImageRunner(cfg.experiment.output_dir)
    policy = RandomPolicy(runner.env)
    agent = KitchenAgent(policy)
    for i in range(cfg.experiment.n_episodes):
        runner.reset()
        runner.run(agent, cfg.experiment.n_steps)


if __name__ == "__main__":
    demo()