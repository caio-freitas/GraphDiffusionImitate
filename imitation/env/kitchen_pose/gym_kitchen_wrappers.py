import gymnasium
import gym
import numpy as np

class KitchenPoseWrapper(gym.Wrapper):
    '''
    Wrapper for FrankaKitchen-v1 environment with joint values and end effector position as observations
    '''
    def __init__(self, max_steps=5000, render=True):
        self.env = gymnasium.make(
            'FrankaKitchen-v1',
            tasks_to_complete=['microwave'],
            render_mode='human' if render else 'rgb_array',
            max_episode_steps=max_steps
        )
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def _get_obs(self, obs):
        obs = obs[0]['observation']
        # 'qp' and 'obj_qp'
        qp = obs[0:9] 
        # Skip 9 to 18 because they are velocities
        obj_qp = obs[18:39]
        return np.concatenate((qp, obj_qp, np.zeros(30)))

    def step(self, action):
        # According to https://robotics.farama.org/envs/franka_kitchen/franka_kitchen/

        all = self.env.step(action)
        obs = self._get_obs(all)

        reward = all[1]
        done = all[2]
        # TODO check if "done" is in 2 or 3
        info = all[4]
        info["success"] = done
        print(f"info: {info}, done: {done}")
        return obs, reward, done, info


    def reset(self):
        all = self.env.reset()
        obs = self._get_obs(all)
        return obs
    
    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed=seed)


class KitchenImageWrapper(gym.Wrapper):
    '''
    Wrapper for FrankaKitchen-v1 environment with RGB images as observations
    '''
    def __init__(self, env, render_hw = (256,360)):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 256, 256))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(9,))
        self.render_hw = render_hw

    def step(self, action):
        obs = self.env.step(action)
        return obs

    def reset(self):
        return self.env.reset()

    def render(self, mode='rgb_array'):
        return self.env.render()
    
    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed=seed)