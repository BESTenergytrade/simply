import warnings

import pandas as pd
from pathlib import Path
import csv

import simply.config as cfg
from simply.actor import Order

LARGE_ORDER_THRESHOLD = 2**32
MARKET_MAKER_THRESHOLD = 2**63-1
ASK = +1
BID = -1


class Market:
    """
    Representation of a market. Collects orders, implements a matching strategy for clearing,
    finalizes post-matching.

    If a network and a grid_fee_matrix parameter are both supplied, Market will favour
    grid_fee_matrix.

    This class provides a basic matching strategy which may be overridden.
    """
    def __init__(self, network=None, grid_fee_matrix=None, time_step=None):
        self.orders = pd.DataFrame(columns=Order._fields)

        self.trades = None
        self.matches = []
        self.t_step = time_step
        self.actor_callback = {}
        self.network = network
        self.save_csv = cfg.config.save_csv
        self.csv_path = Path(cfg.config.path)
        self.grid_fee_matrix = grid_fee_matrix
        if network is not None and grid_fee_matrix is None:
            self.grid_fee_matrix = network.grid_fee_matrix
        if self.save_csv:
            match_header = ["time", "bid_id", "ask_id", "bid_actor", "ask_actor", "bid_cluster",
                            "ask_cluster", "energy", "price", 'included_grid_fee']
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

    def reset(self):
        self.matches = []
        self.trades = None
        self.actor_callback = {}

    def accept_order(self, order, order_id=None, callback=None):
        """
        Handle new order.

        Order must have same time step as market, type must be -1 or +1.
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
        if order is None:
            return

        if order.time != self.t_step:
            raise ValueError("Wrong order time ({}), market is at time {}".format(order.time,
                                                                                  self.t_step))
        # Ignore Orders without energy volume
        if order.energy == 0:
            return

        if order.price is None:
            raise ValueError("Wrong order price ({})".format(order.price))

        if order.type not in [-1, 1]:
            raise ValueError("Wrong order type ({})".format(order.type))

        # look up cluster
        if order.cluster is None and self.network is not None:
            cluster = self.network.node_to_cluster.get(order.actor_id)
            order = order._replace(cluster=cluster)

        # make certain energy has step size of energy_unit
        energy = (
            (order.energy + cfg.config.EPS) // cfg.config.energy_unit) * cfg.config.energy_unit
        # make certain enough energy is traded
        if energy < cfg.config.energy_unit:
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
            self.orders = pd.concat(
                [self.orders, pd.DataFrame([order], dtype=object)],
                ignore_index=True
            )
        else:
            if order_id in self.orders.index:
                raise ValueError("Order ID ({}) already exists".format(order_id))
            new_order = pd.DataFrame([order], dtype=object, index=[order_id])
            self.orders = pd.concat([self.orders, new_order], ignore_index=False)
        self.actor_callback[order.actor_id] = callback
        self.append_to_csv([order], 'orders.csv')

    def clear(self, reset=True):
        """
        Clear market. Match orders, call callbacks of matched orders, reset/tidy up dataframes.

        :param reset: not retaining orders for next market cycle
        :return: None
        """
        # TODO match bids
        matches = self.match(show=cfg.config.show_plots)
        self.matches.append(matches)

        for match in matches:
            bid_actor_callback = self.actor_callback[match["bid_actor"]]
            ask_actor_callback = self.actor_callback[match["ask_actor"]]
            energy = match["energy"]
            price = match["price"]
            if bid_actor_callback is not None:
                bid_actor_callback(self.t_step, 1, energy, price)
            if ask_actor_callback is not None:
                ask_actor_callback(self.t_step, -1, energy, price)
        if reset:
            # don't retain orders for next cycle
            self.orders = pd.DataFrame(columns=Order._fields)
        else:
            # remove fully matched orders
            self.orders = self.orders[self.orders.energy >= cfg.config.energy_unit]

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
        # order by price (while previously original ordering is reversed for equal prices)
        # i.e. higher probability of matching for higher ask prices or lower bid prices
        bids = self.get_bids().iloc[::-1].sort_values(["price"], ascending=False)
        asks = self.get_asks().iloc[::-1].sort_values(["price"], ascending=True)
        matches = []
        for ask_id, ask in asks.iterrows():
            for bid_id, bid in bids.iterrows():
                if ask.actor_id == bid.actor_id:
                    continue
                if self.grid_fee_matrix:
                    self.apply_grid_fee(ask, bid)
                if ask.energy >= cfg.config.energy_unit and bid.energy >= cfg.config.energy_unit \
                        and ask.price <= bid.price:
                    # match ask and bid
                    energy = min(ask.energy, bid.energy)
                    ask.energy -= energy
                    bid.energy -= energy
                    self.orders.loc[ask_id] = ask
                    self.orders.loc[bid_id] = bid
                    matches.append({
                        "time": self.t_step,
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

        output = self.add_grid_fee_info(matches)
        self.append_to_csv(output, 'matches.csv')
        return matches

    def append_to_csv(self, data, filename):
        """
        append_to_csv() appends the given data to the specified CSV file.

        :param data: the data to be appended to the file, as a Pandas DataFrame
        :param filename: the name of the file to which data should be appended
        :return: None
        """
        if self.save_csv:
            saved_data = pd.DataFrame(data, dtype=object)
            saved_data.to_csv(self.csv_path / filename, mode='a', index=False, header=False)

    def create_csv(self, filename, headers):
        """
        create_csv() creates a new CSV file with the given filename at the given path with the given
        headers.

        :param filename: the name of the file to be created
        :param headers: a list of strings representing the headers to be written to the file
        :return: None
        """
        with open(self.csv_path / filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def get_grid_fee(self, match=None, bid_cluster=None, ask_cluster=None):
        """
        Returns the grid fee associated with the bid and ask clusters of a given match.

        :param match: a dictionary representing a match, with keys 'bid_cluster' and 'ask_cluster'
        :param bid_cluster: cluster id of ask
        :param ask_cluster: cluster id of bid
        :return: the grid fee associated with the given bid and ask clusters
        """
        if match or match is not None:
            if bid_cluster or ask_cluster:
                warnings.warn('Either pass match OR ("bid_cluster" and "ask_cluster"),'
                              'otherwise only match information is considered')
            # if match is given, data from the match is used. In other cases bid
            bid_cluster = match['bid_cluster']
            ask_cluster = match['ask_cluster']

        if not self.grid_fee_matrix:
            return cfg.config.default_grid_fee
        else:
            if bid_cluster is None or ask_cluster is None:
                warnings.warn("At least one cluster is 'None', returning default grid fee.")

                # default grid fee
                return cfg.config.default_grid_fee
            else:
                return self.grid_fee_matrix[bid_cluster][ask_cluster]

    def add_grid_fee_info(self, matches):
        """
        Takes in a list of matches and returns the same list with an additional field 'grid_fee'
        added to each match dictionary.

        :param matches: a list of dictionaries representing matches, with keys 'bid_cluster' and
            'ask_cluster'
        :return: the input list of matches with the additional field 'grid_fee'
        """
        output = []
        for match in matches:
            match['included_grid_fee'] = self.get_grid_fee(match)
            output.append(match)
        return output

    def apply_grid_fee(self, ask, bid):
        """
        Updates the given ask price by adding the grid fee associated with the given bid and ask
        clusters.

        :param ask: the ask price to be updated
        :param bid: the bid used to determine the grid fee to be applied
        :return: None
        """
        try:
            ask.price += self.grid_fee_matrix[bid.cluster][ask.cluster]
        except TypeError:
            # if an actor has none as cluster, e.g. the market maker, a TypeError will be thrown.
            # use default grid fee in this case.
            ask.price += cfg.config.default_grid_fee
