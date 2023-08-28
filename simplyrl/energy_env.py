import numpy as np
import pandas as pd
import gymnasium as gym


class EnergyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, actor=None, market=None, horizon=24, training=False, energy_unit=0.001):
        super(EnergyEnv, self).__init__()

        # set actor and market in environment
        self.actor = actor
        self.market = market

        # Size of energy unit
        self.energy_unit = energy_unit

        # How many timesteps to observe when training
        self.horizon = horizon
        self.training = training

        self._setup(self.market)

        # Create action space
        self.action_energy_values = np.arange(-13.5, 13.5, self.energy_unit * 5)
        self.action_space = gym.spaces.MultiDiscrete([len(self.action_energy_values)])

        # Create observation space
        self.observation_values = np.arange(-1.4, 1.4, self.energy_unit)
        self.observation_bins = [len(self.observation_values) for _ in range(self.horizon * 2 + 1)]
        self.observation_space = gym.spaces.MultiDiscrete(self.observation_bins)

        # initialize action, reward and observation attributes
        self.action = None
        self.reward = None
        self.observation = None

    def _setup(self, market=None):
        # Training related elements. Either training data is given or created for rl agent
        # Create market, battery and actor df
        self.m = market
        self.battery = self.actor.battery
        self.actor_df = self.actor.data

    def _step(self, action):

        # assign action from action space
        self.action = round(self.action_energy_values[action[0]], 3)

        # calculate reward
        self.reward = self._get_reward(self.action)

        # Generate the observation for the next timestep
        self.observation = self._next_observation()

        return self.observation, self.reward

    def _next_observation(self):
        """Return observation"""
        self.actor.create_prediction()
        # Add predicted schedule and prices to observation
        observation = self.actor.pred[['schedule', 'prices']].values
        observation = self._handle_observation(observation)
        return observation

    def _handle_observation(self, observation):
        # Flatten observation and add soc to it
        observation = np.append(observation.flatten(), self.actor.battery.soc)
        # Convert observation to discrete values
        observation = np.asarray(np.round((observation + 1.4) * 1000), dtype=np.int64)
        return observation

    def _get_reward(self, action):
        # TODO: Give reward for beating the rule based agent

        # Punish for not operating within limitations:
        battery_energy = self.actor.battery.energy()
        available_energy = battery_energy + action + self.actor.pred['pv'][0]
        needed_energy = self.actor.pred['load'][0]

        if available_energy < needed_energy:
            return self._normalize_reward(-5, 10, -5)
        if available_energy > self.actor.battery.capacity:
            return self._normalize_reward(-5, 10, -5)

        current_reward = self._process_matches()
        if current_reward != self._normalize_reward(-5, 10, -5):
            return self._normalize_reward(self.actor.bank, 10, -5)
        return current_reward

    def _process_matches(self):
        if len(self.m.matches[-1]) > 0:
            for match in self.m.matches[-1]:
                if 'bid_actor' in match and match['bid_actor'] == 'RL':
                    amount_spent = match['price'] * match['energy']
                    self.actor.bank -= amount_spent
                    return self._normalize_reward(-amount_spent, 0.868, -13)

                if 'ask_actor' in match and match['ask_actor'] == 'RL':
                    amount_gained = match['price'] * match['energy']
                    self.actor.bank += amount_gained
                    return self._normalize_reward(amount_gained, max_reward=0.868, min_reward=-3)

    def _normalize_reward(self, raw_reward, max_reward, min_reward):
        # Calculate the range of raw reward values
        reward_range = max_reward - min_reward

        # Normalize the raw reward value using min-max scaling
        normalized_reward = (raw_reward - min_reward) / reward_range

        return normalized_reward

    def render(self, mode='human'):
        pass

    def reset(self):
        self._setup(self.actor_df)
        # observation consists of schedule and market maker buy prices
        observation = pd.concat([self.actor.pred["schedule"], self.actor.mm_buy_prices], axis=1).values
        observation = self._handle_observation(observation)
        return observation
