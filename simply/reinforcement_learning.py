import gym


class SimplyEnv(gym.Env):
    def __init__(self):
        self.action_space = None
        self.observation_space = None
        ...

    def step(self, action):
        ...

    def reset(self):
        ...

    def render(self):
        ...

    def close(self):
        ...
