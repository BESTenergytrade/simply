import pandas as pd
import pymarket
import random


class Market:
    def __init__(self):
        # TODO tbd if lists or dicts or ... is used
        self.asks = []
        self.bids = []
        self.matches = {20: 21}

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
        # TODO send notification to actors about cleared bids

    def match(self):
        # TODO default match can be replaced in different subclass
        bids = self.get_bids()
        asks = self.get_asks()
        # TODO expand bids/asks of identical energy quantity
        # bids.apply(lambda x: [1 for i in range(0, int(x['energy']))], axis=1, result_type='expand')
        bid_ids = list(bids.index)
        ask_ids = list(asks.index)
        bid_ids = random.sample(bid_ids, len(ask_ids))
        map = {b: a for b, a in zip(bid_ids, ask_ids)}
        matches = map
        return matches
