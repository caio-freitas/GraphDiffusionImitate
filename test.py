'''
Script to test a specific policy in an enviroment
Usage: python test.py --config-name=train
'''

import logging
import pathlib

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("eval", eval)

@hydra.main(
        version_base=None,
        config_path=str(pathlib.Path(__file__).parent.joinpath('imitation','config')), 
        config_name="test"
        )
def test(cfg):
    print(OmegaConf.to_yaml(cfg))
    log.info("Running test...")
    # instanciate environment runner from cfg file
    runner = hydra.utils.instantiate(cfg.task.env_runner)
    # instanciate policy from cfg file
    policy = hydra.utils.instantiate(cfg.policy)
    # instanciate agent from policy
    agent = hydra.utils.instantiate(cfg.agent, policy=policy, env=runner.env)

    # run policy in environment
    for i in range(cfg.num_episodes):
        runner.reset()
        runner.run(agent, cfg.max_steps)


if __name__ == "__main__":
    test()