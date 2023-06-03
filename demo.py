"""
Usage:
Training:
python demo.py --multirun +experiment=demo,demo2
"""
from omegaconf import DictConfig, OmegaConf
from imitation.env_runner.kitchen_pose_runner import KitchenPoseRunner
from imitation.policy.random_policy import RandomPolicy
import hydra
import pathlib

@hydra.main(
        version_base=None,
        config_path=str(pathlib.Path(__file__).parent.joinpath('imitation','config'))
        )
def demo(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    runner = KitchenPoseRunner(cfg.experiment.output_dir)
    policy = RandomPolicy(runner.env)
    for i in range(cfg.experiment.n_episodes):
        runner.reset()
        runner.run(policy, cfg.experiment.n_steps)
    runner.run(policy, 100)


if __name__ == "__main__":
    demo()