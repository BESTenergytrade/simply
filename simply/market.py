import pandas as pd
import random


class Market:
    def __init__(self, time):
        # TODO tbd if lists or dicts or ... is used
        self.t = time
        self.asks = []
        self.bids = []
        self.asks_df = None
        self.bids_df = None
        self.matches = {}
        self.energy_unit = 0.1
        self.actor_callback = {}
        self.seed = 42

    def get_bids(self):
        return pd.DataFrame(self.bids, columns=["time", "actor_id", "energy", "price"])

    def get_asks(self):
        return pd.DataFrame(self.asks, columns=["time", "actor_id", "energy", "price"])

    def get_all_matches(self):
        return self.matches

    def print(self):
        print(self.get_bids())
        print(self.get_asks())

    def accept_bid(self, bid, callback):
        """
        :param bid: tuple
        :param callback: callback function
        :return:
        """
        assert bid[0] == self.t
        self.bids.append(bid)
        self.actor_callback[bid[1]] = callback

    def accept_ask(self, ask, callback):
        """
        :param ask: tuple
        :param callback: callback function
        :return:
        """
        assert ask[0] == self.t
        self.asks.append(ask)
        self.actor_callback[ask[1]] = callback

    def clear(self):
        # TODO match bids
        self.matches = self.match()
        # Send cleared bids and asks to actors for further processing via callback
        for a_id, ac in self.actor_callback.items():
            # TODO simplify with one order book
            energy = self.bids_df[self.bids_df["actor_id"] == a_id].count()[0] * self.energy_unit
            price = self.bids_df[self.bids_df["actor_id"] == a_id]["price"].mean() * self.energy_unit
            if energy and price:
                ac(self.t, energy, price)
            energy = - self.asks_df[self.asks_df["actor_id"] == a_id].count()[0] * self.energy_unit
            price = self.asks_df[self.asks_df["actor_id"] == a_id]["price"].mean() * self.energy_unit
            if energy and price:
                ac(self.t, energy, price)

    def match(self, show=False):
        random.seed(self.seed)
        # TODO default match can be replaced in different subclass
        bids = self.get_bids()
        asks = self.get_asks()
        # Expand bids/asks to fixed energy quantity bids/asks with individual ids
        self.bids_df = (
            bids.reindex(bids.index.repeat(bids["energy"] / self.energy_unit))
            .drop("energy", axis=1)
            .reset_index()
        )
        self.asks_df = (
            asks.reindex(asks.index.repeat(asks["energy"] / self.energy_unit))
            .drop("energy", axis=1)
            .reset_index()
        )
        bid_ids = list(self.bids_df.index)
        ask_ids = list(self.asks_df.index)
        if show:
            print(self.bids_df)
            print(self.asks_df)

        # Do the actual matching
        bid_ids = random.sample(bid_ids, min(len(ask_ids), len(bid_ids)))
        matches = {b: a for b, a in zip(bid_ids, ask_ids)}

        return matches
