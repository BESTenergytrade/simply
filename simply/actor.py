import numpy as np
import pandas as pd


class Actor:
    def __init__(self, actor_id, df):
        # TODO add battery component
        # TODO add individual scale values and scale time series accordingly
        self.id = actor_id
        self.t = 0
        self.battery = None
        self.load = df["load"].to_list()
        self.schedule = df["load"].to_list()
        self.prices = df["prices"].to_list()
        self.bids = []
        self.traded = []

    def generate_bid(self):
        # TODO calculate amount of energy to fulfil personal schedule
        energy = self.schedule[self.t]
        # TODO simulate strategy: manipulation, etc.
        price = self.prices[self.t]
        # TODO take flexibility into account to generate the bid

        new_bid = (self.t, self.id, energy, price)
        # TODO place bid on market
        self.bids.append(new_bid)
        self.t += 1

    def receive_market_results(self, energy, price):
        # TODO update schedule, if possible e.g. battery
        # TODO post settlement of differences
        post = (energy, price)
        self.traded.append(post)

    def to_dict(self):
        return self.id


def create_random(actor_id):
    # TODO improve random signals
    # TODO add random generation signal (e.g. with PV characteristic)
    nb_ts = 24
    time_idx = pd.date_range("2021-01-01", freq="H", periods=nb_ts)
    values = np.random.rand(nb_ts, 2)
    cols = ["load", "prices"]
    df = pd.DataFrame(values, columns=cols, index=time_idx)

    return Actor(actor_id, df)
