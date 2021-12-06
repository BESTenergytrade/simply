import pandas as pd
import random
from typing import Dict

import simply.config as cfg
from simply.actor import Order


class Market:
    def __init__(self, time, network=None):
        self.orders = pd.DataFrame(columns=Order._fields).rename_axis('id')
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

    def get_order_dict(self):
        asks = self.get_asks()
        bids = self.get_bids()

        data = {
            "market_1": {
                "bids": bids.to_dict('records'),
                "offers": asks.to_dict('records')
            }
        }

        return data

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

    def define_order_id(self):
        # add final order id to book (remove outdated order id if existent)
        self.orders = self.orders.drop('id', axis=1, errors='ignore')
        self.orders = self.orders.rename_axis('id').reset_index()

    def clear(self, reset=True):
        self.define_order_id()
        if reset:
            assert (self.orders['time'] == self.t).all()

        # Match bids and asks
        matches = self.match(
            self.get_order_dict(),
            self.energy_unit,
            show=cfg.config.show_plots
        )
        self.matches.append(matches)

        # Send actors information about traded volume and price
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

    def match(self, data, energy_unit=0.1, show=False):
        """
        pay as bid. First come, first served
        default match can be replaced in different subclass

        :param data: (Dict[str, Dict]) in format: {"market_name": {{'bids': []], 'offers': []}}
        :param energy_unit: minimal value that can be traded
        :param show: (Bool), print final matches
        :return: matches: (Dict) matched orders respectively
        """
        # only a single market is expected
        assert len(data.items()) == 1
        # bids = pd.DataFrame(data.get(list(data.keys())[0]).get("bids"))
        # asks = pd.DataFrame(data.get(list(data.keys())[0]).get("offers"))
        bids = data.get(list(data.keys())[0]).get("bids")
        asks = data.get(list(data.keys())[0]).get("offers")
        if len(asks) == 0 or len(bids) == 0:
            # no asks or bids at all: no matches
            return {}
        # keep track of unmatched orders (currently only for debugging purposes)
        # orders = pd.concat([pd.DataFrame(bids), pd.DataFrame(asks)]).set_index('id')

        matches = []
        for ask in asks:
            for bid in bids:
                if ask["energy"] >= energy_unit and bid["energy"] >= energy_unit and ask["price"]\
                        <= bid["price"]:
                    # match ask and bid
                    energy = min(ask["energy"], bid["energy"])
                    ask["energy"] -= energy
                    bid["energy"] -= energy
                    # TODO: unmatched orders are not updated/ returned
                    # orders.loc[ask["id"]] = ask
                    # orders.loc[bid["id"]] = bid
                    assert bid["time"] == ask["time"]
                    matches.append({
                        "time": bid["time"],
                        "bid_actor": bid["actor_id"],
                        "ask_actor": ask["actor_id"],
                        "energy": energy,
                        "price": bid["price"]
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
