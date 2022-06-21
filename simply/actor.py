import random

import numpy as np
import pandas as pd
from collections import namedtuple
import matplotlib.pyplot as plt

from simply.util import daily, gaussian_pv
import simply.config as cfg

Order = namedtuple("Order", ("type", "time", "actor_id", "cluster", "energy", "price"))
Order.__doc__ = """
Struct to hold order

:param type: sign of order, representing bid (-1) or ask (+1)
:param time: timestamp when order was created
:param actor_id: ID of ordering actor
:param energy: sum of energy needed or provided. Will be rounded down according to the market's energy unit
:param price: bidding/asking price for 1 kWh
"""

class Actor:
    """
    Actor is the representation of a prosumer, i.e. is holding resources (load, photovoltaic/PV)
    and defining an energy management schedule, generating bids or asks and receiving trading results.

    :param int actor_id: unique identifier of the actor
    :param pandas.DataFrame() df: DataFrame, column names "load", "pv" and "prices" are processed
    :param str csv: Filename in which this actor's data should be stored
    :param float ls: (optional) Scaling factor for load time series
    :param float ps: (optional) Scaling factor for photovoltaic time series
    :param dict pm: (optional) Prediction multiplier used to manipulate prediction time series based on the data time series

    Members:

    id : str
        Identifier of the actor to be set on creation
    grid_id : str
        [unused] Location of the actor in the network (init default: None)
    t : int
        Actor's current time slot should equal current market time slot (init default: 0)
    horizon : int
        [unused] Horizon to which energy management is considered (cfg.parser.get("actor", "horizon", fallback=24))
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
        Assumption of generation and load time series as would be predicted (default: init df + error)
    csv_file : str
        Filename in which this actor's data should be stored
    self.orders : list
        List of generated orders
    self.traded : dict
        Dictionary of received trading results per time slot including matched energy and clearing prices
    """
    def __init__(self, actor_id, df, csv=None, ls=1, ps=1.5, pm={}):
        """
        Actor Constructor that defines an ID, and extracts resource time series from the given
         DataFrame scaled by respective factors as well as the schedule on which basis orders
         are generated.
        """
        # TODO add battery component
        self.id = actor_id
        self.grid_id = None
        self.t = 0
        self.horizon = cfg.parser.get("actor", "horizon", fallback=24)

        self.load_scale = ls
        self.pv_scale = ps
        self.error_scale = 0
        self.battery = None
        self.data = pd.DataFrame()
        self.pred = pd.DataFrame()
        if csv is not None:
            self.csv_file = csv
        else:
            self.csv_file = f'actor_{actor_id}.csv'
        for column, scale in [("load", ls), ("pv", ps), ("prices", 1)]:
            self.data[column] = scale * df[column]
            try:
                prediction_multiplier = np.array(pm[column])
            except KeyError:
                prediction_multiplier = self.error_scale * np.random.rand(self.horizon)
                pm[column] = prediction_multiplier.tolist()
            self.pred[column] = self.data[column].iloc[self.t: self.t + self.horizon] \
                + prediction_multiplier

        if "schedule" in df.columns:
            self.pred["schedule"] = df["schedule"]
        else:
            # perfect foresight
            self.pred["schedule"] = self.pred["pv"] - self.pred["load"]
        self.orders = []
        self.traded = {}
        self.args = {"id": actor_id, "df": df.to_json(), "csv": csv, "ls": ls, "ps": ps,
                     "pm": pm}

    def plot(self, columns):
        """
        Plot columns from an actor's asset data and prediction with suffix label.

        :param str columns: name of the asset that should be plotted
        """
        pd.concat(
            [self.pred[columns].add_suffix("_pred"), self.data[columns]], axis=1
        ).plot()
        plt.show()

    def generate_order(self):
        """
        Generate new order for current time slot according to predicted schedule
        and both store and return it.

        :return: generated new order
        :rtype: Order
        """
        # TODO calculate amount of energy to fulfil personal schedule
        energy = self.pred["schedule"][self.t]
        # TODO simulate strategy: manipulation, etc.
        price = self.pred["prices"][self.t]
        # TODO take flexibility into account to generate the bid

        # TODO replace order type by enum
        new = Order(np.sign(energy), self.t, self.id, None, abs(energy), price)
        self.orders.append(new)
        self.t += 1

        return new

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
        # Cleared market should not be in the past and sign can only take two values
        assert time < self.t
        assert sign in [-1, 1]
        # append traded energy and price to actor's trades
        post = (sign * energy, price)
        pre = self.traded.get(time, ([], []))
        self.traded[time] = tuple(e + [post[i]] for i, e in enumerate(pre))

    def to_dict(self, external_data=False):
        """
        Builds dictionary for saving. external_data returns simple data instead of
        member dump.

        :param dict external_data: (optional) Dictionary with additional data e.g. on prediction error time series
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

        :param str dirpath: (optional) Path of the directory in which the actor csv file should be stored.
        """
        # TODO if "predicted" values do not equal actual time series values,
        #  also errors need to be saved
        if self.error_scale != 0:
            raise Exception('Prediction Error is not yet implemented!')
        save_df = pd.concat([self.data[["load", "pv"]],
                             self.pred[["schedule", "prices"]]], axis=1)
        save_df.to_csv(dirpath.joinpath(self.csv_file))


def create_random(actor_id, start_date="2021-01-01", nb_ts=24, ts_hour=1):
    """
    Create actor instance with random asset time series and random scaling factors

    :param str actor_id: unique actor identifier
    :param str start_date: Start date "YYYY-MM-DD" of the DataFrameIndex for the generated actor's asset time series
    :param int nb_ts: number of time slots that should be generated
    :param ts_hour: number of time slots per hour, e.g. 4 results in 15min time slots
    :return: generated Actor object
    :rtype: Actor
    """
    time_idx = pd.date_range(start_date, freq="{}min".format(int(60/ts_hour)), periods=nb_ts)
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
    ps = random.choices([0, ps], [1-pv_prob, pv_prob], k=1)
    df["schedule"] = ps * df["pv"] - ls * df["load"]
    max_price = 0.3
    df["prices"] *= max_price
    # Adapt order price by a factor to compensate net pricing of ask orders
    # (i.e. positive power) Bids however include network charges
    net_price_factor = 0.7
    df["prices"] = df.apply(
        lambda slot: slot["prices"] - (slot["schedule"] > 0) * net_price_factor
        * slot["prices"], axis=1
    )

    return Actor(actor_id, df, ls=ls, ps=ps)
