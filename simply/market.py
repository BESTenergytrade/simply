import pandas as pd
import random

import simply.config as cfg


class Market:
    def __init__(self, time):
        # TODO tbd if lists or dicts or ... is used
        self.t = time
        self.asks = []
        self.bids = []
        self.orders = []
        self.order_df = None
        self.trades = None
        self.matches = {}
        self.energy_unit = cfg.parser.getfloat("market", "energy_unit", fallback=0.1)
        self.actor_callback = {}

    def get_bids(self):
        self.get_order_df()
        return self.order_df[self.order_df["type"] == 1]

    def get_asks(self):
        self.get_order_df()
        return self.order_df[self.order_df["type"] == -1]

    def get_order_df(self):
        self.order_df = pd.DataFrame(self.orders)
        return self.order_df

    def get_all_matches(self):
        return self.matches

    def print(self):
        print(pd.DataFrame(self.bids))
        print(pd.DataFrame(self.asks))

    def accept_order(self, order, callback):
        """
        :param order: namedtuple namedtuple("Order", ("type", "time", "actor_id", "energy", "price"))
        :param callback: callback function
        :return:
        """
        assert order.time == self.t
        self.orders.append(order)
        if order.type == -1:
            self.bids.append(order)
        elif order.type == 1:
            self.asks.append(order)
        else:
            raise ValueError
        self.actor_callback[order.actor_id] = callback

    def clear(self):
        # TODO match bids
        self.matches = self.match()
        trades = self.order_df.iloc[
            list(self.matches.keys()) + list(self.matches.values())
        ]
        self.trades = trades
        # Send cleared bids and asks to actors for further processing via callback
        for a_id, ac in self.actor_callback.items():
            # TODO simplify with one order book
            a_trades = trades[trades["actor_id"] == a_id]
            if not a_trades.empty:
                assert (a_trades["type"].iloc[0] == a_trades["type"]).all()
                energy = a_trades.count()[0] * self.energy_unit
                price = a_trades["price"].mean() * self.energy_unit
                # TODO replace cleared values by namedtuple
                ac(self.t, a_trades["type"].iloc[0], energy, price)

    def match(self, show=False):
        # TODO default match can be replaced in different subclass
        orders = self.get_order_df()
        # Expand bids/asks to fixed energy quantity bids/asks with individual order ids
        self.order_df = (
            orders.reindex(orders.index.repeat(orders["energy"] / self.energy_unit))
            .drop("energy", axis=1)
            .reset_index()
        )
        bid_ids = list(self.order_df[self.order_df["type"] == 1].index)
        ask_ids = list(self.order_df[self.order_df["type"] == -1].index)
        if show:
            print(self.order_df)

        # Do the actual matching
        bid_ids = random.sample(bid_ids, min(len(ask_ids), len(bid_ids)))
        matches = {b: a for b, a in zip(bid_ids, ask_ids)}

        return matches
