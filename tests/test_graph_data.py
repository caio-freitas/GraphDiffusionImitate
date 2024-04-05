import hydra
from hydra import compose, initialize
import pytest

from omegaconf import OmegaConf


@pytest.fixture
def lift_wrapper():
    # initialize hydra
    with initialize(config_path="../imitation/config"):
        # load the config file
        cfg = compose(config_name="task/lift_graph", overrides=["+pred_horizon=1", 
                                                                "+action_horizon=1",
                                                                "+obs_horizon=1",
                                                                "+max_steps=100",
                                                                "+render=False",
                                                                "+output_video=False",])
        # open hydra config file and instance the dataset
        print(OmegaConf.to_yaml(cfg))
        wrapper = hydra.utils.instantiate(cfg.task.env_runner.env)
        return wrapper

@pytest.fixture
def lift_dataset():
    # import pdb; pdb.set_trace()
    # initialize hydra
    with initialize(config_path="../imitation/config"):
    # load the config file
        cfg = compose(config_name="task/lift_graph", overrides=["+pred_horizon=1", 
                                                                "+action_horizon=1",
                                                                "+obs_horizon=1",])
        # open hydra config file and instance the dataset
        print(OmegaConf.to_yaml(cfg))
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        return dataset


def test_lift_edge_data_match(lift_wrapper, lift_dataset):
    # check if the dataset has the right number of episodes
    
    env_obs = lift_wrapper.reset()
    G_0 = lift_dataset[0]
    assert (env_obs.edge_index == G_0.edge_index).all()
    assert (env_obs.edge_attr == G_0.edge_attr).all()


def test_lift_node_data_match(lift_wrapper, lift_dataset):
    env_obs = lift_wrapper.reset()
    G_0 = lift_dataset[0]
    assert (env_obs.x[:,-1] == G_0.x[:,0,-1]).all()
    assert (env_obs.y.shape == G_0.y[:,0,:].shape)
    
