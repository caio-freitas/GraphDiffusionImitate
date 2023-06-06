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