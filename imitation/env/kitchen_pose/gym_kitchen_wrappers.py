import gymnasium
import gym

class KitchenPoseWrapper(gym.Wrapper):
    '''
    Wrapper for FrankaKitchen-v1 environment with joint values and end effector position as observations
    '''
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(10,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(9,))
        self.env = gymnasium.make(
            'FrankaKitchen-v1',
            tasks_to_complete=['microwave'],
            render_mode='human'
        )

    def step(self, action):
        all = self.env.step(action)
        obs = all[0]['observation']
        reward = all[1]
        done = all[2]
        # TODO check if "done" is in 2 or 3
        info = all[4]
        return obs, reward, done, info


    def reset(self):
        return self.env.reset()

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