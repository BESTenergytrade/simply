import numpy as np
import pandas as pd
from collections import namedtuple
import matplotlib.pyplot as plt

from simply.util import gaussian_pv
import simply.config as cfg

"""
Struct to hold order

type: sign of order, representing bid (-1) or ask (+1)
time: timestamp when order was created
actor_id: ID of ordering actor
energy: sum of energy needed or provided. Will be rounded down according to the market's energy unit
price: bidding/asking price for 1 kWh
"""
Order = namedtuple("Order", ("type", "time", "actor_id", "cluster", "energy", "price"))


class Actor:
    def __init__(self, actor_id, df, csv=None, ls=1, ps=1.5, pm={}):
        """
        Actor is the representation of a prosumer with ressources (load, photovoltaic)

        :param actor_id: unique identification of the actor
        :param df: DataFrame, column names "load", "pv" and "prices" are processed
        :param ls: (optional)
        :param ps: (optional)
        :param pm: (optional)
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
        Plot columns from an actor's predicted schedule.
        """
        pd.concat(
            [self.pred[columns].add_suffix("_pred"), self.data[columns]], axis=1
        ).plot()
        plt.show()

    def generate_order(self):
        """
        Generate new order for current timestep according to predicted schedule.
        Returns order.
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
        Callback function when order is matched. Updates the actor's traded info.
        """
        # TODO update schedule, if possible e.g. battery
        # TODO post settlement of differences
        # cleared market is in the past
        assert time < self.t
        assert sign in [-1, 1]
        # append traded energy and price to actor's trading info
        post = (sign * energy, price)
        pre = self.traded.get(time, ([], []))
        self.traded[time] = tuple(e + [post[i]] for i, e in enumerate(pre))

    def to_dict(self, external_data=False):
        """
        Builds dictionary for saving. external_data returns simple data instead of
        member dump.
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
        Saves data and pred dataframes to given file.
        """
        # TODO if "predicted" values do not equal actual time series values,
        #  also errors need to be saved
        if self.error_scale != 0:
            raise Exception('Prediction Error is not yet implemented!')
        save_df = pd.concat([self.data[["load", "pv"]],
                             self.pred[["schedule", "prices"]]], axis=1)
        save_df.to_csv(dirpath.joinpath(self.csv_file))


def create_random(actor_id):
    """
    Create actor instance with random dataframes .
    """
    # TODO improve random signals
    nb_ts = 24
    time_idx = pd.date_range("2021-01-01", freq="H", periods=nb_ts)
    cols = ["load", "pv", "schedule", "prices"]
    values = np.random.rand(nb_ts, len(cols))
    df = pd.DataFrame(values, columns=cols, index=time_idx)

    # Multiply random generation signal with gaussian/PV-like characteristic
    day_ts = np.linspace(0, 24, 24)
    df["pv"] *= gaussian_pv(day_ts, 12, 3)

    # Scale generation, load and price time series
    ls = 0.7
    ps = 1.5
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
