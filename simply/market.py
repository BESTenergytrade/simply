import pandas as pd
import random

import simply.config as cfg
from simply.actor import Order

class Market:
    """
    Representation of a market. Collects orders, implements a matching strategy for clearing, finalizes post-matching.

    This class provides a basic matching strategy which may be overridden.
    """

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
        # Get all open bids in market. Returns dataframe.
        return self.orders[self.orders["type"] == -1]

    def get_asks(self):
        # Get all open asks in market. Returns dataframe.
        return self.orders[self.orders["type"] == 1]

    def print(self):
        # Debug: print bids and asks to terminal.
        print(self.get_bids())
        print(self.get_asks())

    def accept_order(self, order, callback):
        """
        Handle new order.

        Order must have same timestep as market, type must be -1 or +1.
        Energy is quantized according to the market's energy unit (round down).
        Signature of callback function: matching time, sign for energy direction (opposite of order type), matched energy, matching price.

        :param order: Order (type, time, actor_id, energy, price)
        :param callback: callback function (called when order is successfully matched)
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
        self.orders = pd.concat([self.orders , pd.DataFrame([order])], ignore_index=True)
        self.actor_callback[order.actor_id] = callback

    def clear(self, reset=True):
        """
        Clear market. Match orders, call callbacks of matched orders, reset/tidy up dataframes.
        """
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
        """
        Example matching algorithm: pay as bid, first come first served.

        Return structure: each match is a dict and has the following items:
            time: current market time
            bid_actor: ID of bidding actor
            ask_actor: ID of asking actor
            energy: matched energy (multiple of market's energy unit)
            price: matching price

        This is meant to be replaced in subclasses.
        :param show: show or print plots (mainly for debugging)
        :return: list of dictionaries with matches
        """
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
                        "bid_actor": bid.actor_id,
                        "ask_actor": ask.actor_id,
                        "energy": energy,
                        "price": bid.price
                    })

        if show:
            print(matches)

        return matches

    def save_matches(self, filename='matches.csv'):
        # save all matches in market as dataframe to file
        matches_df = pd.concat(
            [pd.DataFrame.from_dict(self.matches[i]) for i in range(len(self.matches))]
        ).reset_index()
        matches_df.to_csv(filename)

        return matches_df
