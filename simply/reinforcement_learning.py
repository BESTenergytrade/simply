from stable_baselines3 import PPO

import pandas as pd
import gym
from gym import spaces
import numpy as np

from simply.actor import Actor, Order
from simply.market_fair import BestMarket
from simply.power_network import PowerNetwork
import simply.config as cfg
import networkx as nx


class MyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # Training related elements
        self.action_space = gym.spaces.Box(low=np.array([0, 0.01]), high=np.array([1, 1]),
                                           dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(24, 2), dtype=np.float32)

        # Sainity check
        print(f'action {self.action_space.sample()}')
        print(f'observation {self.observation_space.sample()}')

        # Simply related elements
        #         self.horizon = horizon
        self.m = BestMarket(0)
        self.actor = Actor('RL', actor_df, horizon=24)
        print(f'actor time', self.actor.t)
        print(f'market time', self.m.t)

    def _next_observation(self):
        """Return predicted load and prices."""
        # TODO: observation doesn't roll and therefore will include missing values
        observation = self.actor.pred[['load', 'prices']].values
        # replace missing values with 0
        observation = np.nan_to_num(observation, nan=0)
        return observation

    def _take_action(self, action):
        """Parse order from action and accept order into the market."""
        energy = action[0]
        price = action[1]
        if energy < self.actor.pred['load'][0]:
            done = True
            return done
        order = Order(-1, self.actor.t, self.actor.id, self.actor.cluster, energy, price)
        self.m.accept_order(order)
        return False

    def _update_market(self):
        """Add market maker order, clear market and increment timestep"""
        # Add market maker order

        # Match orders, clear market and increment timestep
        self.m.clear()
        self.m.t += 1
        self.actor.t += 1

        # Check if we are out of timesteps
        if self.m.t > self.actor.horizon:
            done = True
            return done
        else:
            done = False
            return done

    def _get_reward(self, action):
        price = action[1]
        return -price

    def step(self, action):
        # Generate order, add to market with market maker order and clear market
        done = self._take_action(action)
        done = self._update_market()

        # Generate the observation for the next timestep
        observation = self._next_observation()
        reward = self._get_reward(action)
        return observation, reward, done, {}

    def reset(self):
        # Reset market and actor
        self.m = BestMarket(0)
        self.actor = Actor('RL', actor_df, horizon=24)
        # Return observation
        observation = self.actor.pred[['load', 'prices']].values
        return observation

    def render(self, action, observation, mode='human'):
        print('------')
        print(f'action: {action}')
        print(f'observation: {observation}')
        print('------')

    def close(self):
        pass


if __name__ == '__main__':
    df = pd.DataFrame(np.random.rand(24, 4), columns=["load", "pv", "prices", "schedule"])
    cfg.Config("")
    nw = nx.Graph()
    nw.add_edges_from([(0, 1, {"weight": 1}), (1, 2), (1, 3), (0, 4)])
    pn = PowerNetwork("", nw, weight_factor=1)
    test_prices = [0.2116079447, 0.1473127859, 0.22184087530000002, 0.11761082760000001,
                   0.2463169965, 0.2020745841, 0.0613031114, 0.24701460990000002,
                   0.12690570210000002, 0.1467477666, 0.0910571313, 0.1510937983, 0.0961995166,
                   0.16232426160000002, 0.1911430976, 0.2395885052, 0.1161007245, 0.1912644558,
                   0.08394693780000001, 0.031559975000000004, 0.07516904740000001, 0.0839614066,
                   0.1340712662, 0.1921131123]

    test_schedule = [-0.2278688066, -0.4956801147, -0.5660800508, -0.4605807878, -0.7235523078,
                     -0.41539310830000004, -0.0517064662, -0.4741886065, -0.253179973,
                     -0.7055324580000001, -0.0665372924, -0.33647962400000003, -0.3992714075,
                     -0.4354996278, -0.625752089, -0.30241824170000003, -0.23024240310000002,
                     -0.6122942333, -0.1880810302, -0.1261036003, -0.18803270630000002,
                     -0.2284269156, -0.7287319187, -0.0596583833]
    example_df = pd.DataFrame(list(zip([abs(num) for num in test_schedule],
                                       test_prices)),
                              columns=['load', 'prices'])
    actor_df = pd.DataFrame(list(zip([abs(num) for num in test_schedule],
                                     [0 for i in range(len(test_schedule))], test_schedule,
                                     test_prices)),
                            columns=['load', 'pv', 'schedule', 'prices'])

    env = MyEnv()
    # env.action_space.sample()

    model = PPO("MlpPolicy", env, verbose=1)
    # Train the agent
    model.learn(total_timesteps=25000)
