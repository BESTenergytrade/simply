import numpy as np
import pandas as pd
from collections import namedtuple
import matplotlib.pyplot as plt

Order = namedtuple("Order", ("type", "time", "actor_id", "energy", "price"))


class Actor:
    def __init__(self, actor_id, df, ls=1, ps=1):
        # TODO add battery component
        self.id = actor_id
        self.t = 0
        self.horizon = 24
        self.load_scale = ls
        self.pv_scale = ps
        self.battery = None
        self.data = pd.DataFrame()
        self.pred = pd.DataFrame()
        self.data["load"] = ls * df["load"]
        self.data["pv"] = ps * df["pv"]
        self.data["prices"] = df["prices"]
        self.pred["load"] = self._predict_horizon(ls * df["load"])
        self.pred["pv"] = self._predict_horizon(ps * df["pv"])
        self.pred["prices"] = self._predict_horizon(df["prices"])
        # perfect foresight
        self.pred["schedule"] = self.pred["pv"] - self.pred["load"]
        self.orders = []
        self.traded = {}

    def _predict_horizon(self, series, n = 0.1):
        return series.iloc[self.t:self.t+self.horizon] + n * np.random.rand(self.horizon)

    def plot(self, columns):
        pd.concat([self.pred[columns].add_suffix("_pred"), self.data[columns]], axis=1).plot()
        plt.show()

    def generate_order(self):
        # TODO calculate amount of energy to fulfil personal schedule
        energy = self.pred["schedule"][self.t]
        # TODO simulate strategy: manipulation, etc.
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
        assert time < self.t
        assert sign in [-1, 1]
        post = (sign * energy, price)
        self.traded[time] = post

    def to_dict(self):
        return self.id


def create_random(actor_id):
    # TODO improve random signals
    # TODO add random generation signal (e.g. with PV characteristic)
    nb_ts = 24
    time_idx = pd.date_range("2021-01-01", freq="H", periods=nb_ts)
    cols = ["load", "pv", "prices"]
    values = np.random.rand(nb_ts, len(cols))
    df = pd.DataFrame(values, columns=cols, index=time_idx)

    return Actor(actor_id, df)
