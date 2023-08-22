import numpy as np
import gymnasium as gym
from simply.actor import Actor, Order
from simply.battery import Battery
from simply.market_fair import BestMarket, MARKET_MAKER_THRESHOLD


class MyEnergyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, horizon=24, initial_soc=None, energy_unit=0.001, stored_value=0.7, actor_df=None, gap=False,
                 training=False):
        super(MyEnergyEnv, self).__init__()
        self.gap = gap
        self.actor_df = actor_df

        # How many timesteps to observe when training
        self.horizon = horizon
        self.training = training

        # Size of energy unit
        self.energy_unit = energy_unit
        self.stored_value = stored_value

        # Initial soc of battery (should be random when training)
        if initial_soc is None:
            self.initial_soc = round(np.random.random(), 3)
        else:
            self.initial_soc = initial_soc

        self._setup(actor_df)

        # Create action space
        self.action_energy_values = np.arange(-13.5, 13.5, self.energy_unit * 5)
        self.action_space = gym.spaces.MultiDiscrete([len(self.action_energy_values)])

        # Create observation space
        self.observation_values = np.arange(-1.4, 1.4, self.energy_unit)
        self.observation_bins = [len(self.observation_values) for _ in range(self.horizon * 2 + 1)]
        self.observation_space = gym.spaces.MultiDiscrete(self.observation_bins)

    def _setup(self, actor_df=None):
        # Training related elements
        if self.actor_df is not None:
            self.actor_df = self.actor_df
        else:
            self.actor_df = create_training_data(self.horizon * 2)

        # Create market, battery and actor
        self.m = BestMarket(0, grid_fee_matrix=[[0]])
        self.m.energy_unit = self.energy_unit
        self.battery = Battery(soc_initial=self.initial_soc)
        self.actor = Actor('RL', self.actor_df, battery=self.battery, horizon=self.horizon, ps=1)

    def step(self, action):
        # Generate order, add to market with market maker order and clear market
        self.previous_soc = self.actor.battery.soc
        self.action = round(self.action_energy_values[action[0]], 3)
        done, problem, order = self._take_action(self.action)

        if not self.training:
            if problem == 'too_little':
                # print('automatic too small')
                self.action = abs(self.actor.pred.schedule[0])
                done, problem, order = self._take_action(self.action)

            if problem == 'too_much':
                # print('automatic too big')
                self.action = -(self.actor.battery.capacity / 2) + self.actor.pred.load[0] + 0.001
                done, problem, order = self._take_action(self.action)

        self.reward = self._get_reward(self.action)
        # Generate the observation for the next timestep
        self.observation = self._next_observation()
        if self.reward == -3:
            done = True

        return self.observation, self.reward, done, {"previous_soc": self.previous_soc, "order": order}

    def _take_action(self, energy):
        """Parse order from action and accept order into the market."""
        # Check if we are out of timesteps
        if self.m.t >= self.actor.horizon:
            done = True
            return done, None, None

        # Check if battery can handle energy
        done, problem = self._check_battery(energy)

        if done:
            return True, problem, None

        # Using the action generate an order and add to the market
        order = self._generate_order(energy)

        # Generate and process other market actors orders
        self._handle_market(order)
        return False, None, order

    def _generate_order(self, energy):
        """Convert energy and price into a simply order."""
        order_type = -1 if energy > 0 else 1
        return Order(order_type, self.actor.t, self.actor.id, 0, abs(energy),
                     self.actor.pred['prices'][0])

    def _check_battery(self, energy):
        """Check if battery can handle energy."""
        # Add energy just bought to battery energy and remove energy needed for current timestep
        total_energy_available = energy + self.actor.battery.energy() + self.actor.pred['pv'][0]

        # If we don't have enough energy for the current timestep end the episode and punish
        if total_energy_available < self.actor.pred['load'][0]:
            # print(f'{self.m.t}_check_battery: Not enough energy for current timestep')
            done = True
            return done, 'too_little'

        # If battery would be charged beyond it's capacity end the episode and punish
        if total_energy_available - self.actor.pred['load'][0] > self.actor.battery.capacity:
            # print(f'{self.m.t}_check_battery: Battery would be charged beyond capacity')
            done = True
            return done, 'too_much'

        return False, None

    def _handle_observation(self, observation):
        # Flatten observation and add soc to it
        observation = np.append(observation.flatten(), self.actor.battery.soc)
        # Convert observation to discrete values
        observation = np.asarray(np.round((observation + 1.4) * 1000), dtype=np.int64)
        return observation

    def _next_observation(self):
        """Return observation"""
        self.actor.create_prediction()
        # Add predicted schedule and prices to observation
        observation = self.actor.pred[['schedule', 'prices']].values
        observation = self._handle_observation(observation)
        return observation

    def _handle_market(self, order):
        """Add market maker, clear market and progress timestep."""
        # Add RL order
        self.m.accept_order(order)

        # Add market maker orders

        mm_ask = Order(1, self.actor.t, 'market_maker', None, MARKET_MAKER_THRESHOLD, self.actor.pred['prices'][0])

        if self.gap:
            mm_bid = Order(-1, self.actor.t, 'market_maker', None, MARKET_MAKER_THRESHOLD,
                           self.actor.pred['prices'][0] - (self.actor.pred['prices'][0] / 10))
        else:
            mm_bid = Order(-1, self.actor.t, 'market_maker', None, MARKET_MAKER_THRESHOLD, self.actor.pred['prices'][0])

        self.m.accept_order(mm_bid)
        self.m.accept_order(mm_ask)

        # Match orders, clear market and increment timestep
        self.m.clear()
        self.m.t += 1
        self.actor.t += 1

    def _update_battery(self, energy):
        """Update battery state of charge."""
        # Deposit this remaining energy in the battery
        total_energy_available = energy + self.actor.battery.energy() + self.actor.pred['pv'][0]
        remaining_energy_available = total_energy_available - self.actor.pred['load'][0]
        self.actor.battery.soc = remaining_energy_available / self.actor.battery.capacity
        # If the battery has charge beyond it's capacity end the episode and punish
        if self.actor.battery.soc > 1:
            # print(f'{self.m.t}_update_battery: Battery has charge beyond it\'s capacity')
            return self._normalize_reward(-5, 10, -5)
        # If the battery has negative charge end the episode and punish
        if self.actor.battery.soc < 0:
            # Set battery soc to 0
            # print(f'{self.m.t}_update_battery: Battery has negative charge')
            self.actor.battery.soc = 0
            return self._normalize_reward(-5, 10, -5)

    def _process_matches(self):
        if len(self.m.matches[-1]) > 0:
            for match in self.m.matches[-1]:
                if 'bid_actor' in match and match['bid_actor'] == 'RL':
                    amount_spent = match['price'] * match['energy']
                    self.actor.bank -= amount_spent
                    soc_done = self._update_battery(match['energy'])
                    if soc_done:
                        return self._normalize_reward(-5, 10, -5)
                    return self._normalize_reward(-amount_spent, 0.868, -13)

                if 'ask_actor' in match and match['ask_actor'] == 'RL':
                    amount_gained = match['price'] * match['energy']
                    self.actor.bank += amount_gained
                    soc_done = self._update_battery(-match['energy'])
                    if soc_done:
                        return self._normalize_reward(-5, 10, -5)
                    return self._normalize_reward(amount_gained, max_reward=0.868, min_reward=-3)
        else:
            soc_done = self._update_battery(0)

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
        # if current_reward == 0:
        #     return current_reward

        # if current_reward is None:
        #     current_reward = self._normalize_reward(0, 0.868, -3)
        return current_reward

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
        observation = self.actor.pred[['schedule', 'prices']].values
        observation = self._handle_observation(observation)
        return observation
