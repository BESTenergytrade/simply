import pandas as pd
import random


class Market:
    def __init__(self):
        # TODO tbd if lists or dicts or ... is used
        self.asks = []
        self.bids = []
        self.matches = {}
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

    def accept_bid(self, bid):
        """
        :param bid: tuple
        :return:
        """
        self.bids.append(bid)

    def accept_ask(self, ask):
        """
        :param ask: tuple
        :return:
        """
        self.asks.append(ask)

    def clear(self):
        # TODO match bids
        self.matches = self.match()
        # TODO send notification to actors about cleared bids for further processing

    def match(self, show=False):
        random.seed(42)
        # TODO default match can be replaced in different subclass
        bids = self.get_bids()
        asks = self.get_asks()
        # Expand bids/asks to fixed energy quantity bids/asks with individual ids
        bids = (
            bids.reindex(bids.index.repeat(bids["energy"]))
            .drop("energy", axis=1)
            .reset_index()
        )
        asks = (
            asks.reindex(asks.index.repeat(asks["energy"]))
            .drop("energy", axis=1)
            .reset_index()
        )
        bid_ids = list(bids.index)
        ask_ids = list(asks.index)
        if show:
            print(bids)
            print(asks)

        # Do the actual matching
        bid_ids = random.sample(bid_ids, len(ask_ids))
        matches = {b: a for b, a in zip(bid_ids, ask_ids)}

        return matches
