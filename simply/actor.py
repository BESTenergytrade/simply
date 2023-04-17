import random
import warnings

import numpy as np
import pandas as pd
from collections import namedtuple
import matplotlib.pyplot as plt

from simply.battery import Battery
from simply.util import daily, gaussian_pv
import simply.config as cfg

EPS = 1e-6
Order = namedtuple("Order", ("type", "time", "actor_id", "cluster", "energy", "price"))
Order.__doc__ = """
Struct to hold order

:param type: sign of order, representing bid (-1) or ask (+1)
:param time: timestamp when order was created
:param actor_id: ID of ordering actor
:param energy: sum of energy needed or provided. Will be rounded down according to the market's
    energy unit
:param price: bidding/asking price for 1 kWh
"""


class Actor:
    """
    Actor is the representation of a prosumer, i.e. is holding resources (load, photovoltaic/PV)
    and defining an energy management schedule, generating bids or asks and receiving trading
    results.
    The actor interacts with the market at every time step in a way defined by the actor strategy.
    The actor has to fulfil his schedule needs by buying power. Buying power can be guaranteed by
    placing orders with at least the market maker price, since the market maker is seen as unlimited
    supply. At the start of every time step the actor can place orders to buy or sell energy at
    the current time step with a predicted schedule and market maker price time series as input.
    After matching took place, the (monetary) bank and resulting soc is calculated taking into
    consideration the schedule and the acquired energy of this time step, i.e. bank and soc at the
    end of the time step. Afterwards the time step is increased and a new prediction for the
    schedule and price is generated.

    :param int actor_id: unique identifier of the actor
    :param pandas.DataFrame() df: DataFrame, column names "load", "pv" and "prices" are processed
    :param str csv: Filename in which this actor's data should be stored
    :param float ls: (optional) Scaling factor for load time series
    :param float ps: (optional) Scaling factor for photovoltaic time series
    :param dict pm: (optional) Prediction multiplier used to manipulate prediction time series based
        on the data time series

    Members:

    id : str
        Identifier of the actor to be set on creation
    grid_id : str
        [unused] Location of the actor in the network (init default: None)
    t : int
        Actor's current time slot should equal current market time slot (init default: 0)
    horizon : int
        [unused] Horizon to which energy management is considered
        (default: cfg.parser.get("actor", "horizon", fallback=24))
    load_scale : float
        Scaling factor for load time series (default: init ls)
    pv_scale : float
        Scaling factor for photovoltaic time series (default: init ps)
    error_scale : float
        [unused] Noise scaling factor (default: 0)
    battery : object
        [unused] Representation of a battery (default: None)
    data : pandas.DataFrame()
        Actual generation and load time series as would be measured (default: init df)
    pred : pandas.DataFrame()
        Assumption of generation and load time series as would be predicted
        (default: init df + error)
    csv_file : str
        Filename in which this actor's data should be stored
    self.orders : list
        List of generated orders
    self.traded : dict
        Dictionary of received trading results per time slot including matched energy and clearing
        prices
    """

    def __init__(self, actor_id, df, battery=None, csv=None, ls=1, ps=1.5, pm={}, cluster=None,
                 strategy: int = 0):
        """
        Actor Constructor that defines an ID, and extracts resource time series from the given
         DataFrame scaled by respective factors as well as the schedule on which basis orders
         are generated.
        """
        # TODO add battery component
        self.id = actor_id
        self.grid_id = None
        self.cluster = cluster
        self.t = cfg.parser.getint("default", "start", fallback=0)
        self.horizon = cfg.parser.getint("default", "horizon", fallback=24)

        self.socs = []
        self.load_scale = ls
        self.pv_scale = ps
        self.error_scale = 0
        self.battery = battery
        if self.battery is None:
            self.battery = Battery(capacity=0)
        self.data = pd.DataFrame()
        self.pred = pd.DataFrame()
        self.pm = pd.DataFrame()
        self.strategy = strategy
        if csv is not None:
            self.csv_file = csv
        else:
            self.csv_file = f'actor_{actor_id}.csv'
        for column, scale in [("load", ls), ("pv", ps), ("prices", 1), ("schedule", 1)]:
            self.data[column] = scale * df[column]
            try:
                self.pm[column] = np.array(pm[column])
            except KeyError:
                prediction_multiplier = self.error_scale * np.random.rand(self.horizon)
                self.pm[column] = prediction_multiplier.tolist()

        self.create_prediction()

        self.market_schedule = self.get_default_market_schedule()

        self.orders = []
        self.traded = {}
        self.args = {"id": actor_id, "df": df.to_json(), "csv": csv, "ls": ls, "ps": ps,
                     "pm": pm}

    def get_market_schedule(self):
        """ Generates a market_schedule for the actor which represents the strategy of the actor
        when to buy or sell energy. At the current time step the actor will always buy/ or sell
        this amount even at market maker price.

        :param strategy: Number representing the actor strategy from 1 to 3
        :type strategy: int
        :return: market_schedule with planed amounts of energy buying/selling per time step
        """
        possible_choices = [0, 1, 2, 3]

        if self.strategy not in possible_choices:
            warnings.warn(
                f"Strategy choice: {self.strategy} was not found in the list of possible "
                f"strategies: {possible_choices}. Using default strategy without "
                "planning instead.")
            self.strategy = 0
        elif self.strategy != 0:
            if self.battery is None or self.battery.capacity == 0:
                warnings.warn(
                    f"Strategy choice: {self.strategy} was found but can not be used since "
                    f"the battery capacity is 0 or no battery exists. Using default strategy "
                    f"without planning instead.")
                self.strategy = 0

        strategy = self.strategy

        if strategy == 0:
            self.market_schedule = self.get_default_market_schedule
            return self.market_schedule

        self.market_schedule = self.plan_global_self_supply()
        if self.strategy == 1:
            return self.market_schedule

        self.market_schedule = self.plan_selling_strategy()
        if strategy == 2:
            return self.market_schedule
        self.market_schedule = self.plan_global_trading()
        return self.market_schedule

    def get_default_market_schedule(self):
        return np.array(self.data["schedule"].values)

    def plan_global_self_supply(self):
        """Returns market_schedule where energy needs are covered by the lowest price slots.

        This strategy predicts the soc of the battery. If the soc would drop below 0 the
        market_schedule will get cover the energy need by finding the cheapest time slot.
        :return: market_schedule
        """
        cum_energy = self.pred.schedule.cumsum() + self.battery.soc * self.battery.capacity
        self.market_schedule = np.array([0] * self.horizon).astype(float)
        # Go through the cumulated demands, deducting the demand if we plan on buying energy
        for i, energy in enumerate(cum_energy):
            while energy < 0:
                soc_prediction = np.ones(self.horizon) * self.battery.soc \
                                 + (cum_energy - self.battery.soc * self.battery.capacity) \
                                 / self.battery.capacity
                # Where is the lowest price in between now and when I will need some energy
                # Only check prices where I dont expect a full soc already or
                # the time where the energy is needed
                possible_global_prices = np.ones(self.horizon) * float('inf')
                # prices are set where the soc in not full yet
                # possible_global_prices[(0- EPS<=soc_prediction )* (soc_prediction < 1 + EPS)] = \
                #     self.prices[(0 -EPS<soc_prediction )* (soc_prediction < 1 + EPS)]
                possible_global_prices[(soc_prediction < 1 - EPS)] = \
                    self.pred.prices[(soc_prediction < 1 - EPS)]

                # index for the last inf value between now and energy demand
                last_inf_index = np.argwhere(possible_global_prices[:i + 1] >= float('inf'))
                if len(last_inf_index) == 0:
                    last_inf_index = 0
                else:
                    last_inf_index = last_inf_index.max()
                possible_global_prices[0:last_inf_index] = float('inf')
                # storing energy before that is not possible. only look at prices afterwards
                min_price_index = np.argmin(possible_global_prices[:i + 1])

                # cheapest price for the energy is when the energy is needed --> no storage needed
                if min_price_index == i or last_inf_index >= i:
                    self.market_schedule[i] -= energy
                    cum_energy[i:] -= energy
                    break

                # cheapest price is some time before the energy is needed. Check the storage
                # how much energy can be stored in the battery
                max_soc = min(1, max(0, np.max(soc_prediction[min_price_index:i])))
                max_storable_energy = (1 - max_soc) * self.battery.capacity

                # how much energy can be stored in the battery per time step via c-rate
                max_storable_energy = min(max_storable_energy, self.battery.capacity *
                                          self.battery.max_c_rate / self.time_steps_per_hour)

                # how much energy do i need to store. Energy needs are negative
                stored_energy = min(max_storable_energy, -energy)
                # Reduce the energy needs for the current time step
                energy += stored_energy

                # fix the soc prediction for the time span between buying and consumption
                # soc_prediction[min_price_index:i] += stored_energy / self.battery.capacity
                self.market_schedule[min_price_index] += stored_energy
                # Energy will be bought this time step. Predictions in the future, e.g. after this
                # time step will use the reduced demand for the time steps afterwards
                cum_energy[min_price_index:] += stored_energy

        soc_prediction = np.ones(self.horizon) * self.battery.soc \
            + (cum_energy - self.battery.soc * self.battery.capacity) / self.battery.capacity
        self.predicted_soc = soc_prediction

        return self.market_schedule

    def plan_selling_strategy(self):
        # Find peaks of SOCs above 1.
        #                                 x
        #             (x)               x   x
        #                   x         x      x
        #           x    x     x    x
        # 1 ----- x-------------  x-
        #      x         o
        #   x
        cum_energy_demand = self.pred.schedule.cumsum() + self.market_schedule.cumsum() +\
                            self.battery.soc * self.battery.capacity
        soc_prediction = np.ones(self.horizon) * self.battery.soc \
            + (cum_energy_demand - self.battery.soc * self.battery.capacity) / self.battery.capacity
        for i, energy in enumerate(cum_energy_demand):
            overcharge = energy - self.battery.capacity
            while overcharge > 0:
                possible_prices = self.pred.selling_price.copy()
                possible_prices[soc_prediction < 0 + EPS] = float('-inf')
                # in between now and the peak, the right most/latest Zero soc does not allow
                # reducing the soc before. Energy most be sold afterwards
                zero_soc_indicies = np.where(soc_prediction[:i] < 0 + EPS)
                if np.any(zero_soc_indicies):
                    right_most_peak = np.max(zero_soc_indicies)
                else:
                    right_most_peak = 0
                possible_prices[:right_most_peak] = float('-inf')
                highest_price_index = np.argmax(possible_prices[:i + 1])

                # If the energy is sold at the time of production no storage is needed
                if highest_price_index == i:
                    self.market_schedule[i] -= overcharge
                    cum_energy_demand[i:] -= overcharge
                    break
                # current_soc_after_schedule=((self.battery.soc*self.battery.capacity)
                #                             +self.pred.schedule[0]+self.global_market_buying_plan[0])/self.battery.capacity
                soc_to_zero = min(np.min(soc_prediction[highest_price_index:i + 1]), 1)
                energy_to_zero = soc_to_zero * self.battery.capacity
                sellable_energy = min(energy_to_zero, overcharge)
                self.market_schedule[highest_price_index] -= sellable_energy
                cum_energy_demand[highest_price_index:] -= sellable_energy
                overcharge -= sellable_energy
                soc_prediction = np.ones(self.horizon) * self.battery.soc \
                    + (cum_energy_demand - self.battery.soc * self.battery.capacity) \
                    / self.battery.capacity

        soc_prediction = np.ones(self.horizon) * self.battery.soc \
            + (cum_energy_demand - self.battery.soc * self.battery.capacity) \
            / self.battery.capacity
        self.predicted_soc = soc_prediction

    def plan_global_trading(self):
        """ Strategy to buy energy when profit is predicted by selling the energy later on
               when the flexibility is given"""
        cum_energy_demand = self.pred.schedule.cumsum() + self.market_schedule.cumsum() + \
            self.battery.soc * self.battery.capacity
        soc_prediction = np.ones(self.horizon) * self.battery.soc \
            + (cum_energy_demand - self.battery.soc * self.battery.capacity) / \
            self.battery.capacity
        buy_prices = np.array(self.pred.prices.values)
        sell_prices = np.array(self.pred.selling_price.values)

        # +++++++++++++++++++++++++=
        # handle stored amount
        buy_index = 0
        buying_price = 0
        sellable_amount_of_stored_energy = min(soc_prediction) * self.battery.capacity
        possible_prices = sell_prices.copy()
        possible_prices[:buy_index] = float("-inf")
        sell_indicies = np.argwhere(possible_prices > buying_price)
        if sell_indicies.size > 0:
            sell_indicies = sell_indicies.squeeze(axis=1)
        # If there are possible selling points of energy and there is the possibility of
        # storing energy in between, i.e soc<1
        while sellable_amount_of_stored_energy > 0+EPS:

            found_sell_index = None
            for sell_index in sell_indicies:
                sell_price = possible_prices[sell_index]
                higher_sell_price_indicies = np.argwhere(
                    possible_prices[sell_index + 1:] >= sell_price)
                # highest price found
                if len(higher_sell_price_indicies) == 0:
                    found_sell_index = sell_index
                    break
                else:
                    # there are higher prices. choose the left most higher price index
                    higher_sell_price_index = higher_sell_price_indicies.min()
                lower_buy_price_indicies = np.argwhere(
                    buy_prices[sell_index + 1:] <= sell_price)
                # No buy dips but still higher selling prices
                if len(lower_buy_price_indicies) == 0:
                    continue
                else:
                    # there are lower buy prices onward. choose the left most lower price index
                    lower_buy_price_index = lower_buy_price_indicies.min()
                # There are better selling points in the future
                if higher_sell_price_index < lower_buy_price_index:
                    continue
                # if not then we have the best selling point
                else:
                    found_sell_index = sell_index
                    break
            # find how much energy can be stored in between buying and selling
            storable_energy = max(
                soc_prediction[buy_index:found_sell_index + 1].max() * self.battery.capacity,
                self.battery.capacity)
            assert storable_energy > 0
            self.market_schedule[found_sell_index] -= sellable_amount_of_stored_energy
            cum_energy_demand[found_sell_index:] -= sellable_amount_of_stored_energy
            soc_prediction = np.ones(self.horizon) * self.battery.soc \
                + (cum_energy_demand - self.battery.soc * self.battery.capacity) / \
                self.battery.capacity
            sellable_amount_of_stored_energy = min(soc_prediction) * self.battery.capacity

        # ++++++++++++++++++++
        sorted_buy_indexes = np.argsort(buy_prices)
        for buy_index in sorted_buy_indexes:
            buying_price = buy_prices[buy_index]
            possible_prices = sell_prices.copy()
            possible_prices[:buy_index + 1] = float("-inf")

            sell_indicies = np.argwhere(possible_prices > buying_price)
            if sell_indicies.size > 0:
                sell_indicies = sell_indicies.squeeze(axis=1)
            # If there are possible selling points of energy and there is the possibility of
            # storing energy in between, i.e soc<1
            while sell_indicies.size > 0 and soc_prediction[
                                             buy_index:sell_indicies[0]+1].max() < 1 - EPS:

                found_sell_index = None

                # make sure selling isnt considered for sell_indicies which lie behind an soc==1
                # event. They can not be used, since the battery can not be charged higher.
                socs = np.array(soc_prediction)
                soc_idx_almost_one = np.argwhere(socs[buy_index:] > 1 - EPS) + buy_index
                if soc_idx_almost_one.size > 0:
                    soc_idx_almost_one = soc_idx_almost_one.squeeze(axis=1)
                    left_most_soc_to_buy_index = min(soc_idx_almost_one)
                    # energy cant be stored over an SOC==1 event. Make selling price impossible
                    possible_prices[left_most_soc_to_buy_index:] = float("-inf")

                for sell_index in sell_indicies:
                    sell_price = possible_prices[sell_index]
                    higher_sell_price_indicies = np.argwhere(
                        possible_prices[sell_index + 1:] >= sell_price)
                    # highest price found
                    if len(higher_sell_price_indicies) == 0:
                        found_sell_index = sell_index
                        break
                    else:
                        # there are higher prices. choose the left most higher price index
                        higher_sell_price_index = sell_index + 1 + higher_sell_price_indicies.min()

                    lower_buy_price_indicies = np.argwhere(
                        buy_prices[sell_index + 1:] <= sell_price)
                    # No buy dips but still higher selling prices
                    if len(lower_buy_price_indicies) == 0:
                        continue
                    else:
                        # there are lower buy prices onward. choose the left most lower price index
                        lower_buy_price_index = higher_sell_price_indicies.min()
                    # There are better selling points in the future
                    if higher_sell_price_index < lower_buy_price_index:
                        continue
                    # if not then we have the best selling point
                    else:
                        found_sell_index = sell_index
                        break
                # find how much energy can be stored in between buying and selling
                storable_energy = (1-soc_prediction[buy_index:found_sell_index + 1].max())\
                    * self.battery.capacity
                assert storable_energy > 0
                self.market_schedule[buy_index] += storable_energy
                self.market_schedule[found_sell_index] -= storable_energy
                cum_energy_demand[buy_index:] += storable_energy
                cum_energy_demand[found_sell_index:] -= storable_energy
                soc_prediction = np.ones(self.horizon) * self.battery.soc \
                    + (cum_energy_demand - self.battery.soc * self.battery.capacity)\
                    / self.battery.capacity
                assert 1 >= max(soc_prediction)-EPS
                assert 0 <= min(soc_prediction[:-2])+EPS
        self.predicted_soc = soc_prediction

    def plot(self, columns):
        """
        Plot columns from an actor's asset data and prediction with suffix label.

        :param str columns: name of the asset that should be plotted
        """
        pd.concat(
            [self.pred[columns].add_suffix("_pred"), self.data[columns]], axis=1
        ).plot()
        plt.show()

    def create_prediction(self):
        """Reset asset and schedule predicition horizon to the current planning time step self.t"""
        # ToDo add selling price
        for column in ["load", "pv", "prices", "schedule"]:
            if column in self.data.columns:
                self.pred[column] = \
                    self.data[column].iloc[self.t: self.t + self.horizon].reset_index(drop=True) \
                    + self.pm[column]
        if "schedule" not in self.data.columns:
            self.pred["schedule"] = self.pred["pv"] - self.pred["load"]

    def update_battery(self, _cache=dict()):
        """Update the battery state depending on the currently scheduled planning time step.

        This function needs to be called once per time step to track the energy inside of the
        battery. It takes the planned, i.e. predicted, schedule and changes the battery's SOC
        accordingly.

        :param _cache: cache of function calls, which SHOULD NOT be provided by user
        """
        # _cache keeps track of method calls by storing the last time of the method call at the
        # key of self/object reference. This makes sure that energy is only taken once per time step
        if self not in _cache:
            _cache[self] = self.t
        else:
            error = "Actor used the battery twice in a single time step"
            assert _cache[self] < self.t, error
            _cache[self] = self.t

        # assumes schedule is positive when pv is produced, Assertion error useful during
        # development to be certain
        # ToDo: Remove for release
        assert self.pred.schedule[0] == self.pred.pv[0] - self.pred.load[0]
        # ToDo Make sure that the balance of schedule and bought energy does not charge
        # or discharge more power than the max c rate
        self.battery.charge(self.pred.schedule[0])

    def generate_order(self):
        """
        Generate new order for current time slot according to predicted schedule
        and both store and return it.

        :return: generated new order
        :rtype: Order
        """
        # TODO calculate amount of energy to fulfil personal schedule
        energy = self.pred["schedule"][0]
        if energy == 0:
            return None
        # TODO simulate strategy: manipulation, etc.
        price = self.pred["prices"][0]
        # TODO take flexibility into account to generate the bid

        # TODO replace order type by enum
        new = Order(np.sign(energy), self.t, self.id, self.cluster, abs(energy), price)
        self.orders.append(new)
        return new

    def next_time_step(self):
        """Update actor and schedule and for next time step.

        Should be executed after clearing.
        Changes actor attributes according to the events in the current time step
        The events can be the impact of schedule and trading on the battery soc or the bank / cost
        for the actor in this time step.
        Update the prediction horizon to the current time step."""

        if self.battery and not self.pred.empty:
            self.update_battery()
            self.socs.append(self.battery.soc)
        self.t += 1
        self.create_prediction()

    def receive_market_results(self, time, sign, energy, price):
        """
        Callback function when order is matched. Updates the actor's individual trading result.

        :param str time: time slot of market results
        :param int sign: for energy sold or bought represented by -1 or +1  respectively
        :param float energy: energy volume that was successfully traded
        :param float price: achieved clearing price for the stated energy
        """

        # order time and actor time have to be in sync
        assert time == self.t
        # sign can only take two values
        assert sign in [-1, 1]
        # append traded energy and price to actor's trades
        post = (sign * energy, price)
        pre = self.traded.get(time, ([], []))
        self.traded[time] = tuple(e + [post[i]] for i, e in enumerate(pre))

    def to_dict(self, external_data=False):
        """
        Builds dictionary for saving. external_data returns simple data instead of
        member dump.

        :param dict external_data: (optional) Dictionary with additional data e.g. on prediction
            error time series
        """
        if external_data:
            args_no_df = {
                "id": self.id, "df": {}, "csv": self.csv_file, "ls": self.load_scale,
                "ps": self.pv_scale, "pm": {}
            }
            return args_no_df
        else:
            return self.args

    def save_csv(self, dirpath):
        """
        Saves data and pred dataframes to given directory with actor specific csv file.

        :param str dirpath: (optional) Path of the directory in which the actor csv file should be
            stored.
        """
        # TODO if "predicted" values do not equal actual time series values,
        #  also errors need to be saved
        if self.error_scale != 0:
            raise Exception('Prediction Error is not yet implemented!')
        save_df = self.data[["load", "pv", "schedule", "prices"]]
        save_df.to_csv(dirpath.joinpath(self.csv_file))


def create_random(actor_id, start_date="2021-01-01", nb_ts=24, ts_hour=1):
    """
    Create actor instance with random asset time series and random scaling factors

    :param str actor_id: unique actor identifier
    :param str start_date: Start date "YYYY-MM-DD" of the DataFrameIndex for the generated actor's
        asset time series
    :param int nb_ts: number of time slots that should be generated
    :param ts_hour: number of time slots per hour, e.g. 4 results in 15min time slots
    :return: generated Actor object
    :rtype: Actor
    """
    time_idx = pd.date_range(start_date, freq="{}min".format(int(60 / ts_hour)), periods=nb_ts)
    cols = ["load", "pv", "schedule", "prices"]
    values = np.random.rand(nb_ts, len(cols))
    df = pd.DataFrame(values, columns=cols, index=time_idx)

    # Multiply random generation signal with gaussian/PV-like characteristic
    for day in daily(df, 24 * ts_hour):
        day["pv"] *= gaussian_pv(ts_hour, 3)

    # Random scale factor generation, load and price time series in boundaries
    ls = random.uniform(0.5, 1.3)
    ps = random.uniform(1, 7)
    # Probability of an actor to possess a PV, here 40%
    pv_prob = 0.4
    ps = random.choices([0, ps], [1 - pv_prob, pv_prob], k=1)
    df["schedule"] = ps * df["pv"] - ls * df["load"]
    max_price = 0.3
    df["prices"] *= max_price
    # Adapt order price by a factor to compensate net pricing of ask orders
    # (i.e. positive power) Bids however include network charges
    net_price_factor = 0.7
    df["prices"] = df.apply(lambda slot: slot["prices"] - (slot["schedule"] > 0)
                            * net_price_factor * slot["prices"], axis=1)
    # makes sure that the battery capacity is big enough, even if no useful trading takes place
    # todo randomize if generate order takes care of meeting demands through a market strategy
    battery_capacity = (df["schedule"].cumsum().max()-df["schedule"].cumsum().min())*2
    return Actor(actor_id, df, battery=Battery(capacity=battery_capacity), ls=ls, ps=ps)


def create_from_csv(actor_id, asset_dict={}, start_date="2021-01-01", nb_ts=None, ts_hour=1,
                    override_scaling=False):
    """
    Create actor instance with random asset time series and random scaling factors. Replace

    :param str actor_id: unique actor identifier
    :param Dict asset_dict: nested dictionary specifying 'csv' filename and column ('col_index')
        per asset or the time series index ('index') of the Actor
    :param str start_date: Start date "YYYY-MM-DD" of the DataFrameIndex for the generated actor's
        asset time series
    :param int nb_ts: number of time slots that should be generated, derived from csv if None
    :param ts_hour: number of time slots per hour, e.g. 4 results in 15min time slots
    :param override_scaling: if True the predefined scaling factors are overridden by the peak value
        of each csv file
    :return: generated Actor object
    :rtype: Actor
    """
    # Random scale factor generation, load and price time series in boundaries
    peak = {
        "load": random.uniform(0.8, 1.3) / ts_hour,
        "pv": random.uniform(1, 7) / ts_hour
    }
    # Probability of an actor to possess a PV, here 40%
    pv_prob = 0.4
    peak["pv"] = random.choices([0, peak["pv"]], [1 - pv_prob, pv_prob], k=1)

    # Initialize DataFrame
    cols = ["load", "pv", "schedule", "prices"]
    df = pd.DataFrame([], columns=cols)

    # Read csv files for each asset
    for col, csv_dict in asset_dict.items():
        # if csv_dict is empty
        if not csv_dict:
            continue
        csv_df = pd.read_csv(
            csv_dict["csv"],
            sep=',',
            parse_dates=['Time'],
            dayfirst=True
        )
        # Rename column and insert data based on dictionary
        df.loc[:, col] = csv_df.iloc[:nb_ts, csv_dict["col_index"]]
        # Override scaling factor by peak value (if True)
        if override_scaling:
            peak[col] = df[col].max()
        # Normalize time series
        df[col] = df[col] / df[col].max()
    # Set new index, in case it was not read in via asset_dict, it is generated
    if "index" not in df.columns:
        df["index"] = pd.date_range(
            start_date,
            freq="{}min".format(int(60 / ts_hour)),
            periods=nb_ts
        )
    df = df.set_index("index")

    # If pv asset key is present but dictionary does not contain a filename
    if "pv" in asset_dict.keys() and not asset_dict["pv"].get("filename"):
        # Initialize PV with random noise
        df["pv"] = np.random.rand(nb_ts, 1)
        # Multiply random generation signal with gaussian/PV-like characteristic per day
        for day in daily(df, 24 * ts_hour):
            day["pv"] *= gaussian_pv(ts_hour, 3)

    # Dummy-Strategy:
    # Predefined energy management, energy volume and price for trades due to no flexibility
    df["schedule"] = peak["pv"] * df["pv"] - peak["load"] * df["load"]
    max_price = 0.3
    df["prices"] = np.random.rand(nb_ts, 1)
    df["prices"] *= max_price
    # Adapt order price by a factor to compensate net pricing of ask orders
    # (i.e. positive power) Bids however include network charges
    net_price_factor = 0.7
    df["prices"] = df.apply(
        lambda slot: slot["prices"] - (slot["schedule"] > 0) * net_price_factor * slot["prices"],
        axis=1)

    return Actor(actor_id, df, ls=peak["load"], ps=peak["pv"])
