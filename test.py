'''
Script to test a specific policy in an enviroment
Usage: python test.py --config-name=train
'''

import hydra
import pathlib
from omegaconf import DictConfig, OmegaConf
import logging


log = logging.getLogger(__name__)

@hydra.main(
        version_base=None,
        config_path=str(pathlib.Path(__file__).parent.joinpath('imitation','config')), 
        config_name="test_pusht_diffusion"
        )
def test(cfg):
    print(OmegaConf.to_yaml(cfg))
    log.info("Running test...")
    # instanciate environment runner from cfg file
    runner = hydra.utils.instantiate(cfg.env_runner)
    # instanciate policy from cfg file
    policy = hydra.utils.instantiate(cfg.policy, env=runner.env)
    # instanciate agent from policy
    agent = hydra.utils.instantiate(cfg.agent, policy=policy)

    # run policy in environment
    for i in range(cfg.num_episodes):
        runner.reset()
        runner.run(agent, cfg.max_steps)


if __name__ == "__main__":
    test()