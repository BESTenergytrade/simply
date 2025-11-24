import warnings
from collections.abc import Iterable

import simply.config as cfg
from simply.market import Market, filter_orders


class MarketMakerDirectTarif(Market):
    """
    MarketMaker direct tarif

    Each timestep, all prosumer actor bids and asks are matched with market maker, respectively.
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
        bids = self.get_bids()
        asks = self.get_asks()

        if len(bids) == 0 or len(asks) == 0:
            # no bids or no asks: no match
            return {}

        # match!
        matches = []

        # filter out market makers (infinite bus) and really large orders
        asks, asks_mm, bids, bids_mm, _, _ = filter_orders(asks, bids)

        # quick exit
        if (asks.empty and bids.empty) \
                or (asks.empty and asks_mm.empty) \
                or (bids.empty and bids_mm.empty):
            # no asks or bids at all: no matches
            return []
        # select one market maker ask and bid
        # pre-filter matchable prosumer orders
        mm_cluster = None
        if not asks_mm.empty:
            mm_cluster = asks_mm.iloc[0].cluster
            asks_mm['price'] += cfg.config.default_grid_fee
            ask_mm_id = asks_mm['price'].astype(float).idxmin()
            ask_mm = asks_mm.loc[ask_mm_id]
        if not bids_mm.empty:
            assert mm_cluster == bids_mm.iloc[0].cluster
            bid_mm_id = bids_mm['price'].astype(float).idxmax()
            bid_mm = bids_mm.loc[bid_mm_id]
            # filter asks
            asks = asks[asks["price"] <= bid_mm.price]

        # Prosumer BUYs (bid) from market maker (ask)
        for bid_id, bid in bids.iterrows():
            # volle Energie der Order wird mit dem Market Maker gematcht
            if bid.energy <= 0:
                continue

            matches.append({
                "time": self.t_step,
                "bid_id": bid_id,
                "ask_id": ask_mm_id,  # Market Maker ask TODO
                "bid_actor": bid.actor_id,
                "ask_actor": ask_mm.actor_id,
                "bid_cluster": bid.cluster,
                "ask_cluster": ask_mm.cluster,
                "energy": bid.energy,
                "price": ask_mm.price,
                "included_grid_fee": cfg.config.default_grid_fee
            })

        # Prosumer SELLs (ask) to market maker (bid)
        for ask_id, ask in asks.iterrows():
            if ask.energy <= 0:
                continue

            matches.append({
                "time": self.t_step,
                "bid_id": bid_mm_id,  # Market Maker bid TODO
                "ask_id": ask_id,
                "bid_actor": bid_mm.actor_id,
                "ask_actor": ask.actor_id,
                "bid_cluster": bid_mm.cluster,
                "ask_cluster": ask.cluster,
                "energy": ask.energy,
                "price": ask.price,  # bid_mm.price,
                "included_grid_fee": 0,
            })

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

        if cfg.config.debug and bid_cluster != ask_cluster:
            warnings.warn('"bid_cluster" and "ask_cluster" are not equal.\n'
                          'Pay-as-Clear Market ignores clusters. '
                          f'Single, fixed grid fee will be used: {self.grid_fee_matrix}')

        return self.grid_fee_matrix
