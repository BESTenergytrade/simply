""" JuH: the script "actor.py" represents the different prosumer,
where only the columns "load", "pv" and "prices" are used. Actor class is defined and can generate orders for the market.
The results are clustered in bids and asks. The pv generation per actor is quantified.
"""


import numpy as np
import pandas as pd
from collections import namedtuple
import matplotlib.pyplot as plt

from simply.util import gaussian_pv



Order = namedtuple("Order", ("type", "time", "actor_id", "energy", "price"))

""" 
JuH: type can be 1 (bid) or -1 (ask)
Time is non-negative timestamp 
actor_id identifies each actor
energy is the sum of energy provided or needed
Price is the asking or bidding price."""

class Actor:
    def __init__(self, actor_id, df, ls=1, ps=2, pm={}):
        """
        JuH: power schedule is implemented to give a foresight for the quantity of energy the prosumer needs to obtain from the market
        and results in the difference of the prediction of the pv load and the load that is needed by the prosumer
        Parameters
        ----------
        actor_id: uique ID
        df: pandas dataframe with load, pv and prices columns
        ls: load scale
        ps: PV scale
        pm: prediction multiplier for each column
         """
        # TODO add battery component
        self.id = actor_id
        self.t = 0
        self.horizon = 24
        self.load_scale = ls
        self.pv_scale = ps
        self.battery = None
        self.data = pd.DataFrame()
        self.pred = pd.DataFrame()
        for column, scale in [("load", ls), ("pv", ps), ("prices", 1)]:
            self.data[column] = scale * df[column]
            try:
                prediction_multiplier = np.array(pm[column])
            except KeyError:
                prediction_multiplier = 0.1 * np.random.rand(self.horizon)
                pm[column] = prediction_multiplier.tolist()
            self.pred[column] = self.data[column].iloc[self.t : self.t + self.horizon] + prediction_multiplier

        # perfect foresight
        self.pred["schedule"] = self.pred["pv"] - self.pred["load"]
        self.orders = []
        self.traded = {}
        self.args = {"id": actor_id, "df": df.to_json(), "ls": ls, "ps": ps, "pm": pm}

    def plot(self, columns):
        """StS: show power predictions of this actor
        """
        pd.concat(
            [self.pred[columns].add_suffix("_pred"), self.data[columns]], axis=1
        ).plot()
        plt.show()

    def generate_order(self):
        """
        StS: generate new Order for the current timestep.
        How much energy do I need/sell and how much am I willing to pay/receive?
        """
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
        """
        StS: callback function which updates the actor's traded property.
        Parameters
        ----------
        time: int
        sign: int
            is devided into bids and asks
        energy: int
        price: int
        """
        # TODO update schedule, if possible e.g. battery
        # TODO post settlement of differences
        # cleared market is in the past
        assert time < self.t
        assert sign in [-1, 1]
        post = (sign * energy, price)
        self.traded[time] = post

    def to_dict(self):
        """StS: returns dictionary with necessary information to recreate this actor from scratch. Orders and trades are not saved.
"""
        return self.args


def create_random(actor_id):
    """
    JuH: creates a Dataframe with a timerange of 24h and random values.
    "util.py" is included into the Dataframe.
    Parameters
    ----------
    actor_id: int

    Returns
    -------
    Object: random Actor
         random power need, PV and prices
    """
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
