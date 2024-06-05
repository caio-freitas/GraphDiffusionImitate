'''
Script to replay joint positions from the robomimic dataset in the robosuite environment
'''


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


class PlaybackPolicy:
    def __init__(self, dataset):
        self.dataset = dataset
        self.t = 0

    def get_action(self, obs):
        self.t += 1
        return self.dataset[self.t].x[:,:,0].T.numpy()

    def train(self, dataset, num_epochs, model_path, seed):
        self.dataset = dataset
        pass

    def load_nets(self, path):
        pass

@hydra.main(
        version_base=None,
        config_path=str(pathlib.Path(__file__).parent.joinpath('..','config')), 
        config_name="test"
        )
def test(cfg):
    print(OmegaConf.to_yaml(cfg))
    log.info("Running test...")
    # instanciate environment runner from cfg file
    runner = hydra.utils.instantiate(cfg.task.env_runner)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    # instanciate policy from cfg file
    policy = PlaybackPolicy(dataset)
    # instanciate agent from policy
    agent = hydra.utils.instantiate(cfg.agent, policy=policy, env=runner.env)

    # run policy in environment
    for i in range(cfg.num_episodes):
        runner.reset()
        runner.run(agent, cfg.max_steps)


if __name__ == "__main__":
    test()