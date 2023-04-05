import random

import numpy as np
import pandas as pd
from collections import namedtuple
import matplotlib.pyplot as plt

from simply.battery import Battery
from simply.util import daily, gaussian_pv
import simply.config as cfg

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

    def __init__(self, actor_id, df, battery=None, csv=None, ls=1, ps=1.5, pm={}, cluster=None):
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

    # ToDo to be implemented in agent branch
    def get_default_market_schedule(self):
        return None

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
        # ToDo add selling price
        for column in ["load", "pv", "prices", "schedule"]:
            if column in self.data.columns:
                self.pred[column] = \
                    self.data[column].iloc[self.t: self.t + self.horizon].reset_index(drop=True) \
                    + self.pm[column]
        if "schedule" not in self.data.columns:
            self.pred["schedule"] = self.pred["pv"] - self.pred["load"]

    def update(self):
        if self.battery and not self.pred.empty:
            self.get_energy()
            self.socs.append(self.battery.soc)
        self.create_prediction()

    def get_energy(self, _cache=dict()):
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
        # update schedule for next time step
        # not part of generate_next_order, since order generation should not lead to next time step
        # other things have to happen, for example matching, and supply of energy through the market
        self.t += 1
        self.update()

    def receive_market_results(self, time, sign, energy, price):
        """
        Callback function when order is matched. Updates the actor's individual trading result.

        :param str time: time slot of market results
        :param int sign: for energy sold or bought represented by -1 or +1  respectively
        :param float energy: energy volume that was successfully traded
        :param float price: achieved clearing price for the stated energy
        """

        # TODO update schedule, if possible e.g. battery
        # TODO post settlement of differences
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
