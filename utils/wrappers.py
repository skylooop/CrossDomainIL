import gymnasium as gym

class UniversalSeed(gym.Wrapper):
    def __init__(self, seed):
        seeds = self.env.seed(seed)
        self.env.observation_space.seed(seed)
        self.env.action_space.seed(seed)
        return seeds
        