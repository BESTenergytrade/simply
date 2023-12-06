from typing import List
import numpy as np
import pandas as pd
import gymnasium as gym
from sb3_contrib.common.envs import InvalidActionEnvDiscrete


class EnergyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, actor=None, market=None, horizon=24, interval=72, training=False, energy_unit=0.001):
        super(EnergyEnv, self).__init__()

        # set actor and market in environment
        self.actor = actor
        self.market = market

        # Size of energy unit
        self.energy_unit = energy_unit

        # How many timesteps to observe when training
        self.horizon_train = horizon
        self.training_interval = interval
        self.training = training

        # Create action space
        battery_cap = actor.battery.capacity
        self.action_energy_values = np.arange(-battery_cap, battery_cap, self.energy_unit * 5)
        self.action_space = gym.spaces.MultiDiscrete([len(self.action_energy_values)])

        # Create observation space
        # TODO: adjust observation space to be variable to config input
        self.observation_values = np.arange(-1.4, 1.4, self.energy_unit)
        # observation is set to be of length 49 with 24 t_steps schedule, buy prices and one soc value
        self.observation_bins = [len(self.observation_values) for _ in range(24 * 2 + 1)]
        self.observation_space = gym.spaces.MultiDiscrete(self.observation_bins)

        # for training initialize action, reward, bank, battery soc and observation attributes
        if training:
            self.reset_train_env()

    def _setup(self):
        """
        Set up training related elements. Current timestep of outer simply simulation and
        rl model observation attributes are initialized
        """

        # setup observation for prediction
        observation = pd.concat([self.actor.pred["schedule"], self.actor.mm_buy_prices], axis=1).values
        soc = self.actor.battery.soc
        self.observation_pred = self._handle_observation(observation, soc)

        if self.training:
            self.t_step_simply = self.actor.t_step
            # choose random t_step to start training episode
            self.t_step_rl = self.random_t_step()
            self.episode_end = self.t_step_rl + self.horizon_train - 1
            self.observation_range[self.t_step_simply] = self.observation_pred
            if self.t_step_rl in self.observation_range.keys():
                self.observation = self.observation_range[self.t_step_rl]
        else:
            self.t_step_rl = self.actor.t_step
            self.action = None
            self.reward = None
            self.bank = None

    def step(self, action):

        # assign action according to timestep of outer simulation
        self.action = self.actions_simply[self.t_step_rl]
        self.reward = self.rewards_simply[self.t_step_rl]
        self.bank = self.banks_simply[self.t_step_rl]
        self.soc = self.socs_simply[self.t_step_rl]

        # Generate the observation for the next timestep
        self.observation = self.observation_range[self.t_step_rl + 1]

        if self.t_step_rl >= self.episode_end:
            terminated = True
        else:
            terminated = False
        truncated = False

        info = {
            "time_step": self.t_step_rl,
            "action": self.action,
            "current_reward": self.reward,
            "bank": self.bank,
        }

        self.t_step_rl += 1

        return self.observation, self.reward, terminated, truncated, info

    def get_action(self):
        return self.actions_simply[self.t_step_rl]

    def _handle_observation(self, observation, soc):
        # Flatten observation and add soc to it
        observation = np.append(observation.flatten(), soc)
        # Convert observation to discrete values
        # TODO: make conversion variable to settings/input
        observation = np.asarray(np.round((observation + 1.4) * 1000), dtype=np.int64)
        return observation

    def random_t_step(self):
        t_step_min = self.t_step_simply - self.training_interval + 1
        t_step_max = self.t_step_simply - self.horizon_train + 1
        return np.random.choice(list(range(t_step_min, t_step_max)))

    def valid_action_mask(self) -> List[bool]:
        self.possible_actions = np.arange(self.action_space.nvec[0])
        actor_battery_cap = self.actor.battery.capacity
        actor_battery_charge = self.actor.battery.soc * actor_battery_cap
        current_schedule = self.actor.pred.schedule[0]
        available_energy = actor_battery_charge + current_schedule
        min_valid = -available_energy
        max_valid = actor_battery_cap - available_energy
        self.valid_actions = np.where((self.action_energy_values < max_valid) & (self.action_energy_values > min_valid))[0]
        return [action in self.valid_actions for action in self.possible_actions]

    def reset_train_env(self):
        self.action = None
        self.reward = None
        self.bank = None
        self.soc = None
        self.observation = None
        self.actions_simply = {}
        self.rewards_simply = {}
        self.banks_simply = {}
        self.socs_simply = {}
        self.observation_range = {}
        self._setup()

    def render(self, mode='human'):
        pass

    def reset(self, seed=None, options=None):
        self._setup()
        # observation consists of schedule and market maker buy prices
        # if prediction return observation of current time step else return observation of training step
        if options == "prediction":
            observation = self.observation_pred
        else:
            observation = self.observation
        info = {
            "time_step": self.t_step_rl,
            "action": self.action,
            "current_reward": self.reward,
            "bank": self.bank,
        }
        return observation, info
