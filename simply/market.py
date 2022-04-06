import pandas as pd
from pathlib import Path
import csv

import simply.config as cfg
from simply.actor import Order

LARGE_ORDER_THRESHOLD = 2**32
MARKET_MAKER_THRESHOLD = 2**63-1


class Market:
    """
    Representation of a market. Collects orders, implements a matching strategy for clearing,
    finalizes post-matching.

    This class provides a basic matching strategy which may be overridden.
    """

    def __init__(self, time, network=None, grid_fee_matrix=None):
        self.orders = pd.DataFrame(columns=Order._fields)
        self.large_bids = pd.DataFrame(columns=Order._fields)
        self.large_asks = pd.DataFrame(columns=Order._fields)
        self.bids_mm = pd.DataFrame(columns=Order._fields)
        self.asks_mm = pd.DataFrame(columns=Order._fields)
        self.t = time
        self.trades = None
        self.matches = []
        self.energy_unit = cfg.parser.getfloat("market", "energy_unit", fallback=0.1)
        self.actor_callback = {}
        self.network = network
        self.save_csv = cfg.parser.getboolean("default", "save_csv", fallback=False)
        self.csv_path = Path(cfg.parser.get("default", "path", fallback="./scenarios/default"))
        if grid_fee_matrix is not None:
            self.grid_fee_matrix = grid_fee_matrix
        elif network is not None:
            self.grid_fee_matrix = network.grid_fee_matrix
        self.EPS = 1e-10
        if self.save_csv:
            match_header = ("time", "bid_id", "ask_id", "bid_actor", "ask_actor", "bid_cluster",
                            "ask_cluster", "energy", "price")
            self.create_csv('matches.csv', match_header)
            self.create_csv('orders.csv', Order._fields)

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

    def filter_market_maker(self, order, ignore_index=True, index=None):
        # filter out market makers (infinite bus) and really large orders
        if order.energy >= MARKET_MAKER_THRESHOLD:
            if order.type == 1:
                self.asks_mm = pd.concat(
                    [self.asks_mm, pd.DataFrame([order], dtype=object, index=index)],
                    ignore_index=ignore_index)
            elif order.type == -1:
                self.bids_mm = pd.concat(
                    [self.bids_mm, pd.DataFrame([order], dtype=object, index=index)],
                    ignore_index=ignore_index)
        else:
            print("WARNING! large order filtered")
            if order.type == 1:
                self.large_asks = pd.concat(
                    [self.large_asks, pd.DataFrame([order], dtype=object, index=index)],
                    ignore_index=ignore_index)
            elif order.type == -1:
                self.large_bids = pd.concat(
                    [self.large_bids, pd.DataFrame([order], dtype=object, index=index)],
                    ignore_index=ignore_index)

    def accept_order(self, order, order_id=None, callback=None):
        """
        Handle new order.

        Order must have same timestep as market, type must be -1 or +1.
        Energy is quantized according to the market's energy unit (round down).
        Signature of callback function: matching time, sign for energy direction
        (opposite of order type), matched energy, matching price.

        :param order: Order (type, time, actor_id, energy, price)
        :param callback: callback function (called when order is successfully matched)
        :param order_id: (optional) define order ID of the order to be inserted, otherwise
          consecutive numbers are used (if this leads to overriding indices, an IndexError is
          raised)
        :return:
        """
        if order.time != self.t:
            raise ValueError("Wrong order time ({}), market is at time {}".format(order.time,
                                                                                  self.t))
        if order.type not in [-1, 1]:
            raise ValueError("Wrong order type ({})".format(order.type))

        # look up cluster
        if order.cluster is None and self.network is not None:
            cluster = self.network.node_to_cluster.get(order.actor_id)
            order = order._replace(cluster=cluster)

        # make certain energy has step size of energy_unit
        energy = ((order.energy + self.EPS) // self.energy_unit) * self.energy_unit
        # make certain enough energy is traded
        if energy < self.energy_unit:
            return
        order = order._replace(energy=energy)
        # If an order ID parameter is not set,
        #   - raise error if current consecuitve number does not equal the total number of orders
        #   - otherwise ignore index -> consecutive numbers are intact
        # otherwise adopt the ID, while checking it is not already used
        if order_id is None:
            if len(self.orders) != 0 and len(self.orders) - 1 != self.orders.index.max():
                raise IndexError("Previous order IDs were defined externally and reset when "
                                 "inserting orders without predefined order_id.")
            if order.energy >= LARGE_ORDER_THRESHOLD:
                self.filter_market_maker(order, ignore_index=True)
            else:
                self.orders = pd.concat(
                    [self.orders, pd.DataFrame([order], dtype=object)],
                    ignore_index=True)
        else:
            new_order = pd.DataFrame([order], dtype=object, index=[order_id])
            if order_id in self.orders.index:
                raise ValueError("Order ID ({}) already exists".format(order_id))
            if order.energy >= LARGE_ORDER_THRESHOLD:
                self.filter_market_maker(order, ignore_index=False, index=[order_id])
            else:
                self.orders = pd.concat([self.orders, new_order], ignore_index=False)
        self.actor_callback[order.actor_id] = callback
        self.append_to_csv([order], 'orders.csv')

    def clear(self, reset=True):
        """
        Clear market. Match orders, call callbacks of matched orders, reset/tidy up dataframes.
        """
        # TODO match bids
        matches = self.match(show=cfg.config.show_plots)
        mm_matches = self.match_market_maker()
        self.matches = self.matches + matches + mm_matches

        for match in self.matches:
            bid_actor_callback = self.actor_callback[match["bid_actor"]]
            ask_actor_callback = self.actor_callback[match["ask_actor"]]
            energy = match["energy"]
            price = match["price"]
            if bid_actor_callback is not None:
                bid_actor_callback(self.t, 1, energy, price)
            if ask_actor_callback is not None:
                ask_actor_callback(self.t, -1, energy, price)

        if reset:
            # don't retain orders for next cycle
            self.orders = pd.DataFrame(columns=Order._fields)
            self.large_bids = pd.DataFrame(columns=Order._fields)
            self.large_asks = pd.DataFrame(columns=Order._fields)
            self.bids_mm = pd.DataFrame(columns=Order._fields)
            self.asks_mm = pd.DataFrame(columns=Order._fields)
        else:
            # remove fully matched orders
            self.orders = self.orders[self.orders.energy >= self.energy_unit]

    def match(self, show=False):
        """
        Example matching algorithm: pay as bid, first come first served.

        Return structure: each match is a dict and has the following items:
            time: current market time
            bid_id: ID of bid order
            ask_id: ID of ask order
            bid_actor: ID of bidding actor
            ask_actor: ID of asking actor
            bid_cluster: cluster of bidding actor
            ask_cluster: cluster of asking actor
            energy: matched energy (multiple of market's energy unit)
            price: matching price

        This is meant to be replaced in subclasses.
        :param show: show or print plots (mainly for debugging)
        :return: list of dictionaries with matches
        """
        asks, bids = self.get_asks(), self.get_bids()
        # order by price (while previously original ordering is reversed for equal prices)
        # i.e. higher probability of matching for higher ask prices or lower bid prices
        bids = bids.iloc[::-1].sort_values(["price"], ascending=False)
        asks = asks.iloc[::-1].sort_values(["price"], ascending=True)
        matches = []
        for ask_id, ask in asks.iterrows():
            for bid_id, bid in bids.iterrows():
                if ask.actor_id == bid.actor_id:
                    continue
                if ask.energy >= self.energy_unit and bid.energy >= self.energy_unit \
                        and ask.price <= bid.price:
                    # match ask and bid
                    energy = min(ask.energy, bid.energy)
                    ask.energy -= energy
                    bid.energy -= energy
                    self.orders.loc[ask_id] = ask
                    self.orders.loc[bid_id] = bid
                    matches.append({
                        "time": self.t,
                        "bid_id": bid_id,
                        "ask_id": ask_id,
                        "bid_actor": bid.actor_id,
                        "ask_actor": ask.actor_id,
                        "bid_cluster": bid.cluster,
                        "ask_cluster": ask.cluster,
                        "energy": energy,
                        "price": bid.price
                    })

        if show:
            print(matches)

        self.append_to_csv(matches, 'matches.csv')
        return matches

    def match_market_maker(self):
        # match with market maker
        # find unmatched orders
        matches = []
        orders = self.orders[(self.orders["energy"] + self.EPS) > self.energy_unit]
        # match asks only with bid market maker with highest price
        asks = orders[orders.type == 1]
        if not self.bids_mm.empty:
            # select bidding market maker by order ID, that has highest price
            bid_mm_id = self.bids_mm['price'].astype(float).idxmax()
            bid_mm = self.bids_mm.loc[bid_mm_id]
            asks = asks[asks["price"] <= bid_mm.price]
            for ask_id, ask in asks.iterrows():
                matches.append({
                    "time": self.t,
                    "bid_id": bid_mm_id,
                    "ask_id": ask_id,
                    "bid_actor": bid_mm.actor_id,
                    "ask_actor": ask.actor_id,
                    "bid_cluster": bid_mm.cluster,
                    "ask_cluster": ask.cluster,
                    "energy": ask.energy,
                    "price": bid_mm.price
                })

        # match bids only with ask market maker with lowest price
        bids = orders[orders.type == -1]
        if not self.asks_mm.empty:
            # select asking market maker by order ID, that has lowest price
            ask_mm_id = self.asks_mm['price'].astype(float).idxmin()
            ask_mm = self.asks_mm.loc[ask_mm_id]
            # indices of matched bids equal order IDs respectively
            bids = bids[bids["price"] >= ask_mm.price]
            for bid_id, bid in bids.iterrows():
                matches.append({
                    "time": self.t,
                    "bid_id": bid_id,
                    "ask_id": ask_mm_id,
                    "bid_actor": bid.actor_id,
                    "ask_actor": ask_mm.actor_id,
                    "bid_cluster": bid.cluster,
                    "ask_cluster": ask_mm.cluster,
                    "energy": bid.energy,
                    "price": ask_mm.price
                })
        self.append_to_csv(matches, 'matches.csv')
        return matches

    def append_to_csv(self, data, filename):
        if self.save_csv:
            saved_data = pd.DataFrame(data, dtype=object)
            saved_data.to_csv(self.csv_path / filename, mode='a', index=False, header=False)

    def create_csv(self, filename, headers):
        with open(self.csv_path / filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
