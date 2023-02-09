import gym
import numpy as np
from baselines import dqn
from simply.market_fair import BestMarket
from actor import Actor, Order

class SimplyEnv(gym.Env):
    def __init__(self):
        # [energy range, price range]
        self.action_space = gym.spaces.Box(low=[-50, 0], high=[50, 5], shape=(2,), dtype=np.float32)
        self.observation_space = None
        self.m = BestMarket
        self.actor = Actor

    def step(self, action):

        # input order into market
        energy = action[0]
        price = action[1]
        order_type = 1 if energy >= 0 else -1
        order = Order(order_type, self.actor.t, self.actor.id, self.actor.cluster, energy, price)
        self.m.accept_order(order)

        # calculate reward function
        observation = None
        reward = None
        done = False
        return observation, reward, done, _

    def reset(self):
        self.m = BestMarket
        ...

    def render(self):
        ...

    def close(self):
        ...


if __name__ == '__main__':
    env = SimplyEnv()

    # Train the agent
    model = dqn.learn(env)

    while True:
        action, _ model.predict(obs[None])
        observation, reward, done, _ = env.step(action)
