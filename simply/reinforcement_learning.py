from stable_baselines3 import PPO
import pandas as pd
import gym
import numpy as np
import os

from simply.actor import Actor, Order
from simply.battery import Battery
from simply.market_fair import BestMarket, MARKET_MAKER_THRESHOLD
from simply.power_network import PowerNetwork
import simply.config as cfg
import networkx as nx
import datetime
import random

def create_random_dataframe(horizon, max_load=3, max_pv=1.3, max_price=0.8):
    """Create a random DataFrame with load, pv and prices."""
    # Create a random array and update it into a DataFrame
    load = np.round(np.random.uniform(0, max_load, horizon).astype('float16'), 3)
    pv = np.round(np.random.uniform(0, max_pv, horizon).astype('float16'), 3)
    prices = np.round(np.random.uniform(0.001, max_price, horizon).astype('float16'), 3)
    
    df = pd.DataFrame({
        'load': load,
        'pv': pv,
        'prices': prices,
        'schedule': (load - pv),
    })
    
    return df

def generate_aggregated_actor_df(horizon=24, index=None, df=None, path='/Users/emilmargrain/Documents/RLI/sample/aggregated_df.csv'):
    if df is None:
        df = pd.read_csv(path).round(3)
        # df = df.copy()
    # df['index'] = df.index
    # df['index'] = df['index'].astype(int)
    if index is None:
        index = random.randrange(0, len(df) - horizon)
    df = df.iloc[index:horizon+index]

    # limit the values to fit the observation space
    df.loc[:, 'load'] = df['load'].clip(0, 1.399)
    df.loc[:, 'pv'] = df['pv'].clip(0, 1.399)
    df.loc[:, 'prices'] = df['prices'].clip(0, 1)

    df.loc[:, 'schedule'] = df['load'] - df['pv']
    return df

def start_training(algorithm_name, initial_soc=None, horizon=96, pretrained_model=None, restart=False, version=0, energy_unit=0.001, stored_value=0.7, aggregated=False, random=False, max_obv=0.8):
        # Set up training variables
    time_now = datetime.datetime.now()
    time_now = time_now.strftime("%m-%d-%H-%M")

    algorithm_name = f'{algorithm_name}_{time_now}'
    TIMESTEPS = 10000
    models_dir = f"models/{algorithm_name}"
    logdir = "logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Create environment

    env = MyEnv(horizon=horizon, initial_soc=initial_soc, energy_unit=energy_unit, stored_value=stored_value, aggregated=aggregated, random=random, max_obv=max_obv)

    # Restart training from a previous model
    if pretrained_model:
        models_dir = f"models/{algorithm_name}"
        pretrained_model_path = f"models/{pretrained_model}/{str(version)}"
        # algorithm_name = pretrained_model
        model = PPO.load(pretrained_model_path, reset_num_timesteps=restart, tensorboard_log=logdir, verbose=1, env=env)
    else:
            model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir, device='cpu')
    try:
        # Train the agent
        for i in range(1, 100000):
            print(i)
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=restart, tb_log_name=f'{algorithm_name}')
            if i % 2 == 0:
                model.save(f"{models_dir}/{TIMESTEPS*i+version}")
            
    except Exception as e:
        print(f'Error: {e}')
        env = model.get_env()
        env.render()

    return algorithm_name, model


class MyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, horizon=24, initial_soc=None, energy_unit=0.001, stored_value=0.7, actor_df=None, aggregated=True, random=False, max_obv=0.8):
        super(MyEnv, self).__init__()

        self.aggregated = aggregated
        self.random = random
        self.actor_df = actor_df
        self.max_obv = max_obv
        self._setup(horizon, initial_soc, energy_unit, stored_value, actor_df)

        # Create action space
        self.action_energy_values = np.arange(-max_obv, max_obv, self.m.energy_unit)
        self.action_price_values = np.arange(0.001, 0.6, self.m.energy_unit)
        self.action_space = gym.spaces.MultiDiscrete([len(self.action_energy_values), len(self.action_price_values)])

        # Create observation space
        self.observation_values = np.arange(0, max_obv, self.m.energy_unit)
        self.observation_bins = [len(self.observation_values) for _ in range(self.horizon * 3 + 1)]
        self.observation_space = gym.spaces.MultiDiscrete(self.observation_bins)    
        
    def _setup(self, horizon, inital_soc, energy_unit, stored_value, actor_df=None):

        # How many timesteps to observe when training
        self.horizon = horizon
        
        # Size of energy unit
        self.energy_unit = energy_unit
        self.stored_value = stored_value

        # Initial soc of battery (should be random when training)
        if inital_soc is None:
            self.inital_soc = round(np.random.random(), 3)
        else:
            self.inital_soc = inital_soc

        # Training related elements
        if self.actor_df is not None:
            self.actor_df = self.actor_df
        elif self.aggregated:
            if self.random:
                self.actor_df = create_random_dataframe(horizon=self.horizon*2, max_load=0.5, max_pv=0.8, max_price=0.6)
            else:
                df = pd.read_csv('/Users/emilmargrain/Documents/GitHub/simply/total_aggregated.csv')
                self.actor_df = generate_aggregated_actor_df(index=0, horizon=self.horizon*2, df=df)
        else:
            self.actor_df = generate_actor_df(self.horizon*2)

        self.m = BestMarket(0, grid_fee_matrix=[[0]])
        self.m.energy_unit = energy_unit
        battery = Battery(soc_initial=self.inital_soc)
        self.actor = Actor('RL', self.actor_df, battery=battery, horizon=self.horizon, ps=1)
        
        self.current_timestep_prediction = self.actor.pred[['load', 'pv', 'prices']].values
        
    def _convert_observation(self, observation):
        """Convert observation to discrete values"""
        # observation = np.digitize(observation, bins=self.observation_values)
        observation = np.multiply(observation, 1000)
        observation = observation.astype(int)
        return observation

    def _add_soc_to_obs(self, observation):
        """Add soc to observation"""
        observation = np.append(observation, self.actor.battery.soc)
        return observation
    
    def _flatten_obs(self, observation):
        """Flatten observation"""
        return observation.flatten()

    def _next_observation(self):
        """Return observation"""
        # replace missing values with 0
        # observation = np.nan_to_num(observation, nan=0)
        self.actor.create_prediction()
        # Add predicted load, pv and prices to observation
        observation = self.actor.pred[['load', 'pv', 'prices']].values

        # Flatten observation
        observation = self._flatten_obs(observation)

        # Add battery soc
        observation = self._add_soc_to_obs(observation)

        # Convert observation to discrete values
        # TODO: Make sure this is rounding correctly
        observation = self._convert_observation(observation)

        # TODO: Add aggregated data from other market actors

        return observation

    def _handle_market(self, order):
        """Add market maker, clear market and progress timestep."""
        # Add RL order
        self.m.accept_order(order)

        # Add market maker orders
        mm_bid = Order(-1, self.actor.t, 'market_maker', None,
                         MARKET_MAKER_THRESHOLD,
                         self.actor.pred['prices'][0])
        mm_ask = Order(1, self.actor.t, 'market_maker', None,
                       MARKET_MAKER_THRESHOLD,
                       self.actor.pred['prices'][0])

        self.m.accept_order(mm_bid)
        self.m.accept_order(mm_ask)

        # Match orders, clear market and increment timestep
        self.m.clear()
        self.m.t += 1
        self.actor.t += 1


    def _check_battery(self, energy):
        """Check if battery can handle energy."""
        # Add energy just bought to battery energy and remove energy needed for current timestep
        total_energy_available = energy + self.actor.battery.energy() + self.actor.pred['pv'][0]

        # If we don't have enough energy for the current timestep end the episode and punish
        if total_energy_available < self.actor.pred['load'][0]:
            print(f'{self.m.t}_check_battery: Not enough energy for current timestep')
            done = True
            return done, 'too_little'
        
        # If battery would be charged beyond it's capacity end the episode and punish
        if (self.actor.battery.energy() + energy + self.actor.pred['pv'][0]) - self.actor.pred['load'][0] > self.actor.battery.capacity:
            print(f'{self.m.t}_check_battery: Battery would be charged beyond capacity')
            done = True
            return done, 'too_much'

        # Calculate energy after load and pv required for current timestep
        remaining_energy_available = total_energy_available - self.actor.pred['load'][0]

        # If we have more energy than the battery can hold end the episode and punish
        if remaining_energy_available > self.actor.battery.capacity:
            print(f'{self.m.t}_check_battery: More energy than the battery can hold')
            done = True
            return done

        return False, None

    def _update_battery(self, energy):
        """Update battery state of charge."""
        # Deposit this remaining energy in the battery
        # TODO: This needs to be updated to account for the battery's charging rate
        total_energy_available = energy + self.actor.battery.energy() + self.actor.pred['pv'][0]
        remaining_energy_available = total_energy_available - self.actor.pred['load'][0]
        self.actor.battery.soc = remaining_energy_available/self.actor.battery.capacity
        # If the battery has charge beyond it's capacity end the episode and punish
        if self.actor.battery.soc > 1:
            print(f'{self.m.t}_update_battery: Battery has charge beyond it\'s capacity')
            return -10
        # If the battery has negative charge end the episode and punish
        if self.actor.battery.soc < 0:
            # Set battery soc to 0
            print(f'{self.m.t}_update_battery: Battery has negative charge')
            self.actor.battery.soc = 0
            return -10

    def _generate_order(self, energy, price):
        """Convert energy and price into a simply order."""
        order_type = -1 if energy > 0 else 1
        return Order(order_type, self.actor.t, self.actor.id, 0, abs(energy),
                     price)

    def _take_action(self, action):
        """Parse order from action and accept order into the market."""
        # Check if we are out of timesteps
        if self.m.t >= self.actor.horizon:
            done = True
            return done, None

        # Parse action
        energy = action[0]
        price = action[1]

        # Check if battery can handle energy
        done, problem = self._check_battery(energy)
        if done:
            return True, problem, None

        # Using the action generate an order and add to the market
        order = self._generate_order(energy, price)

        # Generate and process other market actors orders
        self._handle_market(order)
        return False, None, order

    def _process_matches(self):
        reward = 0
        if len(self.m.matches[-1]) > 0:
            for match in self.m.matches[-1]:
                # Reward buying low
                if 'bid_actor' in match and match['bid_actor'] == 'RL': 
                    amount_spent = match['price'] * match['energy']
                    # reward -= amount_spent
                    self.actor.bank -= amount_spent
                    soc_done = self._update_battery(match['energy'])
                    # reward += self._get_advanced_reward(amount_spent)
                    if soc_done:
                        print(f'{self.m.t}_process_matches: bid actor soc_done')
                        return -10
                    return amount_spent

                # Reward selling high
                if 'ask_actor' in match and match['ask_actor'] == 'RL': 
                    amount_gained = match['price'] * match['energy']
                    # reward += amount_gained
                    self.actor.bank += amount_gained
                    soc_done = self._update_battery(-match['energy'])
                    # reward += self._get_advanced_reward(amount_gained)
                    if soc_done:
                        print(f'{self.m.t}_process_matches: ask actor soc_done')
                        return -10
                    return amount_gained
        else:
            soc_done = self._update_battery(0)

        return reward

    def _get_battery_value(self, battery_discount=0.5):
        """Return reward based on energy in battery."""
        return (self.actor.battery.energy() * self.actor.pred['prices'].mean()) * battery_discount

    def _get_reward(self, action):
        # Punish for not operating within limitations:
        battery_energy = self.actor.battery.energy()
        available_energy = battery_energy + action[0] + self.actor.pred['pv'][0]
        needed_energy = self.actor.pred['load'][0]

        if available_energy < needed_energy:
            print(f'{self.m.t}_get_reward: available_energy < needed_energy')
            return -10
        if available_energy > self.actor.battery.capacity:
            print(f'{self.m.t}_get_reward: available_energy > self.actor.battery.capacity')
            return -10

        # TODO: improve reward function around profit
        current_reward = self._process_matches()
        if current_reward == -10:
            return current_reward

        # Experiment with the reward being the amount of money in the bank combined with a discounted battery reward
        battery_reward = self._get_battery_value(self.stored_value)

        return ((self.actor.bank + battery_reward) * 10) + current_reward
        # return ((self.actor.bank + battery_reward) * 10)   
        # return self.actor.bank * 10   
        # return current_reward

    def _get_advanced_reward(self, price):
        """Return a reward designed to encourage the agent to buy low and sell high."""

        # Calculate the minimum and maximum values for each column
        mins = self.actor.pred['prices'].min()
        maxs = self.actor.pred['prices'].max()

        # Reward for buying low
        if self.action[0] > 0:
            # Scale each value in the dataframe to be proportionally between 0 and 1
            scaled_price = (price - maxs) / (mins - maxs)
            return scaled_price

        # Reward for selling high
        else:
            # Scale each value in the dataframe to be proportionally between 0 and 1
            scaled_price = (price - mins) / (maxs - mins)
            return scaled_price

    def _convert_action(self, action):
        """Convert action to continuous values."""
        energy = round(self.action_energy_values[action[0]], 3)
        price = round(self.action_price_values[action[1]], 3)
        return [energy, price]

    def step(self, action):
        # Generate order, add to market with market maker order and clear market
        self.current_timestep_prediction = self.actor.pred[['load', 'pv', 'prices']].values
        # Convert action to continuous values
        self.previous_soc = self.actor.battery.soc
        self.action = self._convert_action(action)
        
        # done = self._take_action(self.action)

        failed, problem, order = self._take_action(self.action)
        if problem == 'too_little':
            self.action = [self.actor.pred.load[0], self.actor.pred.prices[0]+0.001]
            failed, problem, order  = self._take_action(self.action)
        if problem == 'too_much':
            # TODO: Experiment with different rules for when the battery is full and the load energy is still being bought
            print('NEW2')
            self.action = [-(self.actor.battery.capacity/2), 0.001]
            failed, problem, order  = self._take_action(self.action)

        self.reward = self._get_reward(self.action)
        # Generate the observation for the next timestep
        self.observation = self._next_observation()
        if self.reward == -10:
            done = True
        else:
            done = False
        return self.observation, self.reward, done, {"previous_soc": self.previous_soc, "order": order}
    
    def back_step(self):
        self.m.t -= 1
        self.actor.t -= 1
        return self.previous_observation

    def reset(self):
        # Reset training related elements
        self._setup(self.horizon, self.inital_soc, self.energy_unit, self.stored_value, self.actor_df)
        
        # Prepare initial observation
        observation = self.actor.pred[['load', 'pv', 'prices']].values
        # Flatten observation
        observation = self._flatten_obs(observation)
        # Add battery soc
        observation = self._add_soc_to_obs(observation)
        # Convert observation to discrete values
        observation = self._convert_observation(observation)
        return observation

    def render(self, mode='human'):
        print('------')
        saved_df = pd.DataFrame(self.actor.pred)
        print(f't: {self.actor.t - 1}')
        print(f'self.actor.pred')
        print(self.current_timestep_prediction)
        print('action: ', self.action)
        print(f'matches: {self.m.matches[-1]}')
        print(f'bank: {self.actor.bank}')
        print(f'after match battery energy: {self.actor.battery.energy()}')
        print(f'after match battery soc: {self.actor.battery.soc}')
        print(f'reward: {self.reward}')
        print(f'observation: {self.observation_values[self.observation]}')


if __name__ == '__main__':
    aggregated=True
    # algorithm, model = start_training('aggregated_complex_reward_day_and_a_half_03-04-09-14', horizon=36, pretrained_model='aggregated_complex_reward_day_and_a_half_03-04-09-14_03-04-12-58', version=850000, aggregated=aggregated)
    # algorithm_name, model = start_training('24/1.4/complex+random2', horizon=24, aggregated=aggregated, random=True, max_obv=1.4)
