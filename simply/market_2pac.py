import warnings
from collections.abc import Iterable

import matplotlib.pyplot as plt

from simply.market import Market, MARKET_MAKER_THRESHOLD
import simply.config as cfg
from simply.market_maker import MARKETMAKERID


class TwoSidedPayAsClear(Market):
    """
    Two sided Pay-As-Clear market mechanism, similar to
    https://gridsingularity.github.io/gsy-e/two-sided-pay-as-clear/

    Each timestep, the highest bids are matched with the lowest offers.
    """

    def __init__(self, network=None, grid_fee_matrix=None, time_step=None):
        if grid_fee_matrix is None:
            warnings.warn("Two sided Pay-As-Clear market was generated without a grid_fee_matrix "
                          "in its constructor. The market will use the grid fee from the "
                          f"configuration for all trades.\n Grid Fee = "
                          f"{cfg.config.default_grid_fee}")
            grid_fee_matrix = cfg.config.default_grid_fee

        assert_error = "grid_fee_matrix must be a single value in case of wo sided Pay-As-Clear " \
                       "markets"
        assert not isinstance(grid_fee_matrix, Iterable), assert_error

        # This will throw an error even if assertions are turned off, if grid_fee_matrix is not a
        # numeric value
        self.grid_fee_matrix = float(grid_fee_matrix)
        super().__init__(network=network, grid_fee_matrix=grid_fee_matrix, time_step=time_step)

    def match(self, show=False):
        # order orders by price
        bids = self.get_bids().sort_values(["price", "energy"], ascending=False)
        asks = self.get_asks().sort_values(["price", "energy"], ascending=True)
        if show:
            plot_merit_order(bids, asks)

        if len(bids) == 0 or len(asks) == 0:
            # no bids or no asks: no match
            return {}

        # match!
        bid_iter = bids.iterrows()
        bid_id, bid = next(bid_iter)
        matches = []
        clearing_price_reached = False
        for ask_id, ask in asks.iterrows():
            if clearing_price_reached:
                break
            while bid is not None:
                if (ask.price + self.grid_fee_matrix > bid.price or
                        ask.actor_id == bid.actor_id == MARKETMAKERID or
                        ask.energy + bid.energy > MARKET_MAKER_THRESHOLD * 2 * 0.9):
                    # Clearing price reached when ask + grid fee > bid price but also when
                    # market maker matches with itself. In case the market maker is not named
                    # properly it might be recognized by the order volume
                    clearing_price_reached = True
                    break
                # get common energy value
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
                    "price": ask.price + self.grid_fee_matrix,
                    "included_grid_fee": self.grid_fee_matrix,
                })
                if bid.energy + cfg.config.EPS < cfg.config.energy_unit:
                    # bid finished: next bid
                    try:
                        bid_id, bid = next(bid_iter)
                    except StopIteration:
                        bid = None
                if ask.energy + cfg.config.EPS < cfg.config.energy_unit:
                    # ask finished: next ask
                    break

        # adjust price to market clearing price (highest asking price)
        for match in matches:
            match["price"] = matches[-1]["price"]

        if show:
            print(matches)

        self.append_to_csv(matches, 'matches.csv')
        return matches

    def get_grid_fee(self, match=None, ask_cluster=None, bid_cluster=None):
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

        if bid_cluster != ask_cluster:
            warnings.warn('"bid_cluster" and "ask_cluster" are not equal.\n'
                          'Pay-as-Clear Market ignores clusters. '
                          f'Single, fixed grid fee will be used: {self.grid_fee_matrix}')

        return self.grid_fee_matrix


def plot_merit_order(bids, asks):
    mm_bid = bids[bids["actor_id"] == MARKETMAKERID]
    mm_ask = asks[asks["actor_id"] == MARKETMAKERID]
    bids = bids[bids["actor_id"] != MARKETMAKERID]
    asks = asks[asks["actor_id"] != MARKETMAKERID]

    # value assignment in iterrows does not change dataframe -> original shown
    bid_x, bid_y = bids["energy"].to_list(), bids["price"].to_list()
    bid_y = [next(iter(bid_y), None)] + bid_y
    bid_x_sum = [0] + [sum(bid_x[:(i + 1)]) for i, _ in enumerate(bid_x)]
    ask_x, ask_y = asks["energy"].to_list(), asks["price"].to_list()
    ask_y = [next(iter(ask_y), None)] + ask_y
    ask_x_sum = [0] + [sum(ask_x[:(i + 1)]) for i, _ in enumerate(ask_x)]

    plt.figure()
    plt.step(bid_x_sum, bid_y, where="pre", label="bids")
    plt.step(ask_x_sum, ask_y, where="pre", label="asks")
    plt.hlines(mm_bid.price, 0, bid_x_sum[-1], colors='b', ls='dashdot', label="mm_bid")
    plt.hlines(mm_ask.price, 0, ask_x_sum[-1], colors='y', ls='dashed', label="mm_bid")
    plt.legend()
    plt.xlabel("volume")
    plt.ylabel("price")
    plt.show()
