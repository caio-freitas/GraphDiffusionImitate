import gymnasium as gym

class KitchenPoseWrapper(gym.Wrapper):
    '''
    Wrapper for FrankaKitchen-v1 environment with joint values and end effector position as observations
    '''
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(10,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,))
        self.env = gym.make(
            'FrankaKitchen-v1',
            tasks_to_complete=['microwave', 'kettle'],
            render_mode='human'
            )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()

    def render(self, mode='human'):
        return self.env.render(mode=mode)

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