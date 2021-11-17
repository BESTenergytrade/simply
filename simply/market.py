import pandas as pd
import random

import simply.config as cfg
from simply.actor import Order

class Market:
    def __init__(self, time, network=None):
        # TODO tbd if lists or dicts or ... is used
        self.orders = pd.DataFrame(columns = Order._fields)
        self.t = time
        self.trades = None
        self.matches = []
        self.energy_unit = cfg.parser.getfloat("market", "energy_unit", fallback=0.1)
        self.actor_callback = {}
        self.network = network

    def get_bids(self):
        return self.orders[self.orders["type"] == -1]

    def get_asks(self):
        return self.orders[self.orders["type"] == 1]

    def print(self):
        print(self.get_bids())
        print(self.get_asks())

    def accept_order(self, order, callback):
        """
        :param order: Order (type, time, actor_id, energy, price)
        :param callback: callback function
        :return:
        """
        if order.time != self.t:
            raise ValueError("Wrong order time ({}), market is at time {}".format(order.time, self.t))
        if order.type not in [-1, 1]:
            raise ValueError("Wrong order type ({})".format(order.type))
        # make certain energy has step size of energy_unit
        energy = (order.energy // self.energy_unit) * self.energy_unit
        # make certain enough energy is traded
        if energy < self.energy_unit:
            return
        self.orders = self.orders.append(pd.DataFrame([order]), ignore_index=True)
        self.actor_callback[order.actor_id] = callback

    def clear(self, reset=True):
        # TODO match bids
        matches = self.match(show=cfg.config.show_plots)
        self.matches.append(matches)

        for match in matches:
            bid_actor_callback = self.actor_callback[match["bid_actor"]]
            ask_actor_callback = self.actor_callback[match["ask_actor"]]
            energy = match["energy"]
            price = match["price"]
            bid_actor_callback(self.t, 1, energy, price)
            ask_actor_callback(self.t,-1, energy, price)

        if reset:
            # don't retain orders for next cycle
            self.orders = pd.DataFrame()
        else:
            # remove fully matched orders
            self.orders = self.orders[self.orders.energy >= self.energy_unit]

    def match(self, show=False):
        # pay as bid. First come, first served
        # default match can be replaced in different subclass

        matches = []
        for ask_id, ask in self.get_asks().iterrows():
            for bid_id, bid in self.get_bids().iterrows():
                if ask.energy >= self.energy_unit and bid.energy >= self.energy_unit and ask.price <= bid.price:
                    # match ask and bid
                    energy = min(ask.energy, bid.energy)
                    ask.energy -= energy
                    bid.energy -= energy
                    self.orders.loc[ask_id] = ask
                    self.orders.loc[bid_id] = bid
                    matches.append({
                        "time": self.t,
                        "bid_actor": int(bid.actor_id),
                        "ask_actor": int(ask.actor_id),
                        "energy": energy,
                        "price": bid.price
                    })

        if show:
            print(matches)

        return matches

    def save_matches(self, filename='matches.csv'):
        matches_df = pd.concat(
            [pd.DataFrame.from_dict(self.matches[i]) for i in range(len(self.matches))]
        ).reset_index()
        matches_df.to_csv(filename)

        return matches_df
