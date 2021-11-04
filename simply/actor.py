import numpy as np
import pandas as pd
from collections import namedtuple
import matplotlib.pyplot as plt

from simply.util import gaussian_pv
import simply.config as cfg

Order = namedtuple("Order", ("type", "time", "actor_id", "energy", "price"))


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
            self.pred[column] = self.data[column].iloc[self.t : self.t + self.horizon] + prediction_multiplier

        # perfect foresight
        self.pred["schedule"] = self.pred["pv"] - self.pred["load"]
        self.orders = []
        self.traded = {}
        self.args = {"id": actor_id, "df": df.to_json(), "csv": csv, "ls": ls, "ps": ps, "pm": pm}

    def plot(self, columns):
        pd.concat(
            [self.pred[columns].add_suffix("_pred"), self.data[columns]], axis=1
        ).plot()
        plt.show()

    def generate_order(self):
        # TODO calculate amount of energy to fulfil personal schedule
        energy = self.pred["schedule"][self.t]
        # TODO simulate strategy: manipulation, etc.
        # TODO replace price update by realistic net price timeseries
        # Quick fix: Adapt order price to compensate due to net pricing of ask orders
        net_price_factor = 0.3
        self.pred["prices"][self.t] -= (energy < 0) * net_price_factor * self.pred["prices"][self.t]
        price = self.pred["prices"][self.t]
        # TODO take flexibility into account to generate the bid

        # TODO replace order type by enum
        new = Order(np.sign(energy), self.t, self.id, abs(energy), price)
        # TODO place bid on market
        self.orders.append(new)
        self.t += 1

        return new

    def receive_market_results(self, time, sign, energy, price):
        # TODO update schedule, if possible e.g. battery
        # TODO post settlement of differences
        # cleared market is in the past
        assert time < self.t
        assert sign in [-1, 1]
        # append traded energy and price to actor's trading info
        post = (sign * energy, price)
        pre = self.traded.get(time, ([], []))
        self.traded[time] = tuple(e + [post[i]] for i,e in enumerate(pre))

    def to_dict(self, external_data=False):
        if external_data:
            args_no_df = {
                "id": self.id, "df": {}, "csv": self.csv_file, "ls": self.load_scale, "ps": self.pv_scale, "pm": {}
            }
            return args_no_df
        else:
            return self.args

    def save_csv(self, dirpath):
        self.data.to_csv(dirpath.joinpath(self.csv_file))


def create_random(actor_id):
    # TODO improve random signals
    nb_ts = 24
    time_idx = pd.date_range("2021-01-01", freq="H", periods=nb_ts)
    cols = ["load", "pv", "prices"]
    values = np.random.rand(nb_ts, len(cols))
    df = pd.DataFrame(values, columns=cols, index=time_idx)

    # Multiply random generation signal with gaussian/PV-like characteristic
    day_ts = np.linspace(0, 24, 24)
    df["pv"] *= gaussian_pv(day_ts, 12, 3)
    return Actor(actor_id, df)
