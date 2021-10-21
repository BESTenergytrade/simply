import pandas as pd
import random

from simply.actor import Order

class Market:
    def __init__(self, time, network=None):
        # TODO tbd if lists or dicts or ... is used
        self.orders = pd.DataFrame()
        self.t = time
        self.matches = []
        self.energy_unit = 0.1
        self.actor_callback = {}
        self.network = network

    def get_bids(self):
        return self.orders[self.orders["type"] == 1]

    def get_asks(self):
        return self.orders[self.orders["type"] == -1]

    def print(self):
        print(self.get_bids())
        print(self.get_asks())

    def accept_order(self, order, callback):
        """
        :param order: namedtuple namedtuple("Order", ("type", "time", "actor_id", "energy", "price"))
        :param callback: callback function
        :return:
        """
        assert order.time == self.t
        assert order.type in [-1, 1]
        # make certain energy has step size of energy_unit
        energy = (order.energy // self.energy_unit) * self.energy_unit
        # make certain enough energy is traded
        if energy < self.energy_unit:
            return
        unit_order = Order(order.type, order.time, order.actor_id, energy, order.price)
        self.orders = self.orders.append(pd.DataFrame([unit_order]), ignore_index=True)
        self.actor_callback[order.actor_id] = callback

    def clear(self, reset=True):
        # TODO match bids
        matches = self.match()
        self.matches += matches

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
        # TODO default match can be replaced in different subclass
        bids = self.get_bids()
        asks = self.get_asks()

        # pay as bid. First come, first served
        matches = []
        for ask_id, ask in asks.iterrows():
            for bid_id, bid in bids.iterrows():
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
