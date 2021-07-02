import pandas as pd
import random

from simply.actor import Order

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
        self.energy_unit = 0.1
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
        # make certain energy has step size of energy_unit
        energy = (order.energy // self.energy_unit) * self.energy_unit
        unit_order = Order(order.type, order.time, order.actor_id, energy, order.price)
        self.orders.append(unit_order)
        if order.type == -1:
            self.bids.append(unit_order)
        elif order.type == 1:
            self.asks.append(unit_order)
        else:
            raise ValueError
        self.actor_callback[order.actor_id] = callback

    def clear(self):
        # TODO match bids
        self.matches = self.match()

        for match in self.matches:
            bid_actor_id = match["bid_actor"]
            ask_actor_id = match["ask_actor"]
            bid_actor_callback = self.actor_callback[bid_actor_id]
            ask_actor_callback = self.actor_callback[ask_actor_id]
            energy = match["energy"]
            price = match["price"]
            bid_actor_callback(self.t, 1, energy, price)
            ask_actor_callback(self.t,-1, energy, price)

    def match(self, show=False):
        # TODO default match can be replaced in different subclass
        orders = self.get_order_df()
        bids = orders[orders["type"] > 0]
        asks = orders[orders["type"] < 0]

        # pay as bid. First come, first served
        matches = []
        for ask_id, ask in asks.iterrows():
            for bid_id, bid in bids.iterrows():
                if ask.energy > 0 and bid.energy > 0 and ask.price <= bid.price:
                    # match ask and bid
                    energy = min(ask.energy, bid.energy)
                    ask.energy -= energy
                    bid.energy -= energy
                    matches.append({
                        "bid_actor": int(bid.actor_id),
                        "ask_actor": int(ask.actor_id),
                        "energy": energy,
                        "price": bid.price
                    })

        if show:
            print(matches)

        return matches
