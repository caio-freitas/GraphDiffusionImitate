"""
Usage:
Training:
python demo.py
"""
import logging
from omegaconf import DictConfig, OmegaConf
from imitation.agent.kitchen_agent import KitchenAgent
from imitation.env_runner.robomimic_lowdim_runner import RobomimicEnvRunner
from imitation.policy.random_policy import RandomPolicy
import hydra
import pathlib

log = logging.getLogger(__name__)

@hydra.main(
        version_base=None,
        config_path=str(pathlib.Path(__file__).parent.joinpath('imitation','config')),
        config_name="demo"
        )
def demo(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    log.info("Running demo...")
    runner = hydra.utils.instantiate(cfg.env_runner)
    policy = hydra.utils.instantiate(cfg.policy, env=runner.env)
    agent = hydra.utils.instantiate(cfg.agent, policy=policy)
    runner.reset()
    runner.run(agent, cfg.max_steps)


if __name__ == "__main__":
    demo()