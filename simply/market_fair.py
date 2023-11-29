import pandas as pd
from simply.market import Market
import simply.config as cfg
from typing import List
from simply.market import LARGE_ORDER_THRESHOLD
from simply.market import MARKET_MAKER_THRESHOLD
from time import time


def time_it(function, timers={}):
    """Decorator function to time the duration and number of function calls.

    :param function: function do be decorated
    :type function: function
    :param timers: storage for cumulated time and call number
    :type timers: dict
    :return: decorated function or timer if given function is None
    :rtype function or dict

    """
    if function == "flush":
        keys = [key for key in timers.keys()]
        for key in keys:
            del timers[key]
        return
    if function:
        def decorated_function(*this_args, **kwargs):
            key = function.__name__
            start_time = time()
            return_value = function(*this_args, **kwargs)
            delta_time = time() - start_time
            try:
                timers[key]["time"] += delta_time
                timers[key]["calls"] += 1
            except KeyError:
                timers[key] = dict(time=0, calls=1)
                timers[key]["time"] += delta_time
            return return_value

        return decorated_function
    sorted_timer = dict(sorted(timers.items(), key=lambda x: x[1]["time"] / x[1]["calls"]))
    return sorted_timer


class BestCluster:
    """Class which keeps track of attributes resolving around a cluster and
    implements functionality to keep the best algorithm more readable.

    A single BestCluster contains the bids of this cluster, the asks which are
    matched at the current state with these bids, a copy of unmatched asks,
    its own clearing price, and offers methods like getting possible profits if
    asks would be inserted"""

    def __init__(self, idx, bestmarket: "BestMarket"):
        self.market: BestMarket = bestmarket
        self.idx = idx
        self.bids: pd.DataFrame = pd.DataFrame()
        self.asks: pd.DataFrame = pd.DataFrame()
        self.matches = []
        self.clearing_price: float = -float("inf")
        self.bid_clearing_price: float = None
        self.clearing_price_reached: bool = False
        self.matched_energy_units: int = 0
        self.ask_iterator = []
        self._row: int

    def __repr__(self):
        return f"BestCluster {self.idx} with matched units: {self.matched_energy_units}, " \
               f"clearing price: {self.clearing_price}"

    @time_it
    def match_locally(self):
        clearing = get_clearing(self.bids, self.asks,
                                prev_clearing_energy=self.matched_energy_units,
                                ask_iterator=self.ask_iterator)
        self.matched_energy_units = clearing["matched_energy_units"]
        self.clearing_price = clearing["clearing_price"]
        self.bid_clearing_price = clearing["bid_clearing_price"]

    @time_it
    def get_insertion_profit(self, ask):
        ask = ask.copy()
        ask.adjusted_price = ask.price + self.market.get_grid_fee(bid_cluster=self.idx,
                                                                  ask_cluster=ask.cluster)
        asks = self.asks.copy()
        asks.loc[ask.name] = ask
        asks = asks.sort_values(["adjusted_price", "price"], ascending=[True, False])
        clearing = get_clearing(self.bids, asks)
        return clearing["clearing_price"] - ask.adjusted_price, clearing

    @time_it
    def remove(self, ask):
        if cfg.config.debug:
            print(f"removing {ask.name} from cluster {self.idx}")
        old_matched_energy = self.matched_energy_units
        self.asks = self.asks.drop(ask.name)
        # removing ask can change clearing. if the amount of energy stays the
        # same, this ask should be removed from other clusters, where it was
        # not matched before
        self.match_locally()
        if old_matched_energy == self.matched_energy_units:
            for cluster in self.market.clusters:
                if cluster == self:
                    continue
                try:
                    cluster.asks = cluster.asks.drop(ask.name)
                    cluster.asks = cluster.asks.drop(ask.name)
                except KeyError:
                    pass

    @time_it
    def insert(self, ask, clearing):

        old_matched_energy = self.matched_energy_units
        ask.adjusted_price = ask.price + self.market.get_grid_fee(
            bid_cluster=self.idx, ask_cluster=ask.cluster)
        self.asks.loc[ask.name] = ask
        self.asks = self.asks.sort_values(["adjusted_price", "price"], ascending=[True, False])
        # use the clearing which was calculated for insertion already
        self.matched_energy_units = clearing["matched_energy_units"]
        self.clearing_price = clearing["clearing_price"]
        self.bid_clearing_price = clearing["bid_clearing_price"]

        if old_matched_energy == self.matched_energy_units:
            # An ask was inserted but the matched energy stayed the same. In other words an old
            # matched ask got removed from matching in this cluster. therefore it becomes available
            # in other clusters
            ask = self.asks.iloc[self.matched_energy_units - 1]
            clusters = [cluster for cluster in self.market.clusters if cluster != self]
            dispute_value = -float("inf")
            best_profit = -float("inf")
            best_clearing, best_cluster, best_profit = \
                self.market.get_best_cluster(dispute_value, best_profit, ask, clusters)
            if (best_profit > 0 or
                    best_profit == 0 and
                    best_clearing["matched_energy_units"] > best_cluster.matched_energy_units):
                # insert the ask if it generates profit or if it at least increases the amount of
                # matched energy at 0 profit
                best_cluster.insert(ask, best_clearing)
            else:
                # best_profit is negative. therefore it will not be matched at this state. all other
                # clusters get this ask, for possible matching
                for cluster in clusters:
                    cluster.asks[ask.name] = ask
                    cluster.asks = cluster.asks.sort_values(["adjusted_price", "price"],
                                                            ascending=[True, False])
        elif old_matched_energy > clearing["matched_energy_units"]:
            # this should never happen
            raise Exception


class BestMarket(Market):
    """
    Custom fair market mechanism.

    Similar to two-sided pay-as-clear, but searches globally for best matches, taking network
    fees into account.Nodes are first grouped into clusters (nodes with no transaction fees
    between them).
    Then, all clusters are evaluated individually, adding transaction fees to other clusters.
    If a match becomes disputed (order matched more than once), the higher offer is taken,
    while the other one is removed as a possible match and that cluster is re-evaluated.
    This converges to an optimal solution.
    """

    def __init__(self, network=None, grid_fee_matrix=None, time_step=None,
                 disputed_matching='bid_price'):
        super().__init__(network, grid_fee_matrix, time_step)
        self.clusters: List[BestCluster] = []
        # ToDo: enum-type would be nicer than string
        self.disputed_matching = disputed_matching

    @time_it
    def resolve_dispute(self, ask, bid_cluster):
        if self.disputed_matching == "grid_fee":
            # for dispute values bigger is better, therefore negative price
            return -self.get_grid_fee(bid_cluster=bid_cluster.idx,
                                      ask_cluster=ask.cluster)
        elif self.disputed_matching == "bid_price":
            try:
                asks = bid_cluster.asks.copy()
                ask = ask.copy()
                ask.adjusted_price = self.get_grid_fee(bid_cluster=bid_cluster.idx,
                                                       ask_cluster=ask.cluster)
                asks[ask.name] = ask
                asks = asks.sort_values(["adjusted_price", "price"], ascending=[True, False])
                ask_row = asks.index.get_loc(ask.name)
                val = bid_cluster.bids.iloc[bid_cluster.ask_iterator.index(ask_row)].price
            except (KeyError, IndexError, ValueError):
                val = -float("inf")
            return val
        raise ValueError

    @time_it
    def clusters_to_match_exist(self):
        for cluster in self.clusters:
            if not cluster.clearing_price_reached:
                # at least on cluster still has not reached its clearing price
                return True
        return False

    def get_clusters_to_match(self):
        return [cluster for cluster in self.clusters if not cluster.clearing_price_reached]

    @time_it
    def match_old(self, show=False):
        asks = self.get_asks()
        bids = self.get_bids()

        # filter out market makers (infinite bus) and really large orders
        asks, asks_mm, bids, bids_mm, large_asks, large_bids = self.filter_orders(asks, bids)

        if (asks.empty and bids.empty) \
                or (asks.empty and asks_mm.empty) \
                or (bids.empty and bids_mm.empty):
            # no asks or bids at all: no matches
            return []

        # filter out orders without cluster (can still be matched with market maker)
        asks = asks[~asks.cluster.isna()]
        bids = bids[~bids.cluster.isna()]

        # split asks and bids into the smallest energy unit, save original index, add cluster idx
        asks = self.split_orders_to_energy_unit(asks)
        bids = self.split_orders_to_energy_unit(bids)

        # keep track which clusters have to be (re)matched
        # start with all clusters
        clusters_to_match = set(range(len(self.grid_fee_matrix)))
        # keep track of matches
        matches = []
        # keep track which asks to exclude in each zone (initially empty)
        exclude = {cluster_idx: set() for cluster_idx in clusters_to_match}

        while clusters_to_match:
            # simulate local market within cluster
            cluster_idx = clusters_to_match.pop()

            # get local bids
            _bids = bids[bids.cluster == cluster_idx]
            if len(_bids) == 0:
                # no bids within this cluster: can't match here
                continue

            # get all asks that are not in exclude list for this cluster (copy asks, retain index)
            _asks = asks.drop(exclude[cluster_idx], axis=0, inplace=False)
            # annotate asking price by grid-fee:
            # get cluster ID for all asks, preserve ordering
            ask_cluster_ids = list(_asks.cluster)
            # get grid_fees from any node in cluster to different ask actors
            grid_fees = self.grid_fee_matrix[cluster_idx]
            # get grid_fees in same order as ask node IDs
            ask_grid_fees = [grid_fees[i] for i in ask_cluster_ids]
            # set adjusted price with network grid-fee
            _asks["adjusted_price"] = pd.Series(_asks.price + ask_grid_fees, index=_asks.index)

            # order local bids and asks by price
            _bids = _bids.sort_values(["price"], ascending=False)
            _asks = _asks.sort_values(["adjusted_price", "price"], ascending=[True, False])

            # match local bids and asks
            bid_iter = _bids.iterrows()
            bid_id, bid = next(bid_iter)
            _matches = []
            for ask_id, ask in _asks.iterrows():
                # compare with next bid
                if bid is not None:
                    # still bids in queue
                    if ask.adjusted_price <= bid.price:
                        # bid and ask match: append to local solution
                        _matches.append({
                            "time": self.t_step,
                            "bid_id": bid_id,
                            "ask_id": ask_id,
                            "bid_actor": bid.actor_id,
                            "ask_actor": ask.actor_id,
                            "bid_cluster": bid.cluster,
                            "ask_cluster": ask.cluster,
                            "energy": cfg.config.energy_unit,
                            "price": ask.adjusted_price,
                            "included_grid_fee": ask.adjusted_price - ask.price
                        })
                    # get next bid
                    try:
                        bid_id, bid = next(bid_iter)
                    except (StopIteration, IndexError):
                        bid = None
                # get next ask

            # remove old matches from same cluster
            matches = [m for m in matches if m["bid_cluster"] != cluster_idx]

            for _match in _matches:
                # adjust price to local market clearing price (highest asking price)
                _match["price"] = _matches[-1]["price"]

                # try to merge into global matches
                # remove double matches
                # asks are removed where market clearing price is lower
                for match_idx, match in enumerate(matches):
                    if match["ask_id"] == _match["ask_id"]:
                        # same ask: compare prices
                        if _match["price"] > match["price"]:
                            # new match is better:
                            # exclude old match
                            exclude[match["bid_cluster"]].add(match["ask_id"])
                            # replace old match
                            matches[match_idx] = _match
                            # redo other cluster
                            clusters_to_match.add(match["bid_cluster"])
                        else:
                            # old match is better: exclude new match
                            exclude[cluster_idx].add(_match["ask_id"])
                            # redo current cluster
                            clusters_to_match.add(cluster_idx)
                        break
                else:
                    # new match does not conflict: insert as-is
                    matches.append(_match)

        matches = self.group_matches(asks, bids, matches)

        # match with market maker
        # find unmatched orders
        orders = self.orders[(self.orders["energy"] + cfg.config.EPS) > cfg.config.energy_unit]
        # ignore large orders
        orders = orders[~orders.index.isin(large_asks.index)]
        orders = orders[~orders.index.isin(large_bids.index)]
        # match asks only with bid market maker with highest price
        asks = orders[orders.type == 1]
        if not bids_mm.empty:
            # select bidding market maker by order ID, that has highest price
            bids_mm['price'] += cfg.config.default_grid_fee
            bid_mm_id = bids_mm['price'].astype(float).idxmax()
            bid_mm = bids_mm.loc[bid_mm_id]
            asks = asks[asks["price"] <= bid_mm.price]
            for ask_id, ask in asks.iterrows():
                matches.append({
                    "time": self.t_step,
                    "bid_id": bid_mm_id,
                    "ask_id": ask_id,
                    "bid_actor": bid_mm.actor_id,
                    "ask_actor": ask.actor_id,
                    "bid_cluster": bid_mm.cluster,
                    "ask_cluster": ask.cluster,
                    "energy": ask.energy,
                    "price": bid_mm.price,
                    "included_grid_fee": cfg.config.default_grid_fee
                })

        # match bids only with ask market maker with lowest price
        bids = orders[orders.type == -1]
        if not asks_mm.empty:
            # select asking market maker by order ID, that has lowest price
            asks_mm['price'] += cfg.config.default_grid_fee
            ask_mm_id = asks_mm['price'].astype(float).idxmin()
            ask_mm = asks_mm.loc[ask_mm_id]
            # indices of matched bids equal order IDs respectively
            bids = bids[bids["price"] >= ask_mm.price]
            for bid_id, bid in bids.iterrows():
                matches.append({
                    "time": self.t_step,
                    "bid_id": bid_id,
                    "ask_id": ask_mm_id,
                    "bid_actor": bid.actor_id,
                    "ask_actor": ask_mm.actor_id,
                    "bid_cluster": bid.cluster,
                    "ask_cluster": ask_mm.cluster,
                    "energy": bid.energy,
                    "price": ask_mm.price,
                    "included_grid_fee": cfg.config.default_grid_fee
                })

        if show:
            print(matches)

        output = self.add_grid_fee_info(matches)
        self.append_to_csv(output, 'matches.csv')

        return matches

    @time_it
    def group_matches(self, asks, bids, matches):
        # group matches: ask -> bid -> match
        _matches = {}
        for match in matches:
            # get original order id of ask/bid and adjust order energy
            ask_id = match["ask_id"]
            ask_order_id = asks.loc[ask_id].order_id
            self.orders.loc[ask_order_id, "energy"] -= match["energy"]
            if cfg.config.debug:
                print(f'Order id {ask_order_id}: {self.orders.loc[ask_order_id, "energy"]}')
            assert self.orders.loc[ask_order_id, "energy"] + cfg.config.EPS >= 0, \
                f"Volume of Order ID {ask_order_id} exceeded, e.g. by matched energy unit ask_id " \
                f"{ask_id}."
            bid_id = match["bid_id"]
            bid_order_id = bids.loc[bid_id].order_id
            self.orders.loc[bid_order_id, "energy"] -= match["energy"]

            if ask_order_id not in _matches:
                # ask not seen before: create empty dict
                _matches[ask_order_id] = dict()
            ask_matches = _matches[ask_order_id]
            try:
                m = ask_matches[bid_order_id]
                # bid has already been matched with this ask: price has to be identical
                assert m["price"] == match["price"]
                m["energy"] += match["energy"]
            except KeyError:
                # new bid was matched
                m = match
                m["ask_id"] = ask_order_id
                m["bid_id"] = bid_order_id
            # update dictionary
            ask_matches[bid_order_id] = m
        # retrieve matches from nested dict
        matches = [m for ask_matches in _matches.values() for m in ask_matches.values()]
        return matches

    @time_it
    def match(self, show=False):
        return self.match_new(show)

    @time_it
    def match_new(self, show=False):
        asks = self.get_asks()
        bids = self.get_bids()

        # filter out market makers (infinite bus) and really large orders
        asks, asks_mm, bids, bids_mm, _, _ = self.filter_orders(asks, bids)

        if (asks.empty and bids.empty) \
                or (asks.empty and asks_mm.empty) \
                or (bids.empty and bids_mm.empty):
            # no asks or bids at all: no matches
            return []

        # filter out actor orders without cluster
        asks = asks[~asks.cluster.isna()]
        bids = bids[~bids.cluster.isna()]

        # split asks and bids into the smallest energy unit, save original index, add cluster idx
        asks = self.split_orders_to_energy_unit(asks)
        bids = self.split_orders_to_energy_unit(bids)

        # Fill the orders up with MM bids/asks, so that MM could potentially match with totality
        # of posted orders.
        # Within each cluster only MarketMaker asks are copied into to be matched,
        # except the MarketMaker Cluster, where those MarketMaker will be filtered out.
        bids_mm.energy = cfg.config.energy_unit * len(asks)
        bids_mm = self.split_orders_to_energy_unit(bids_mm)
        asks_mm.energy = cfg.config.energy_unit * len(bids)
        asks_mm = self.split_orders_to_energy_unit(asks_mm)
        bids_mm.index = bids_mm.index + len(bids)
        asks_mm.index = asks_mm.index + len(asks)
        bids = pd.concat([bids, bids_mm])
        asks = pd.concat([asks, asks_mm])
        # if asks and bids are already equal in volume, there is no

        # keep track which clusters have to be (re)matched
        # start with all clusters

        self.clusters = [BestCluster(idx=idx, bestmarket=self) for idx in
                         list(range(len(self.grid_fee_matrix))) + [None]]

        # Work with copy to be able to remove empty bid clusters
        clusters_to_match = self.get_clusters_to_match()
        for cluster in clusters_to_match:
            # simulate local market within cluster

            # get local bids
            cluster.bids = bids[bids.cluster == cluster.idx]
            cluster.asks = asks.copy()
            if cluster.idx is None:
                # This is the MM cluster with cluster None which does not match cluster.idx
                assert cluster.bids.empty
                # set this Cluster with MM bids
                cluster.bids = bids[bids.cluster.isna()]
                # filter out MM asks
                cluster.asks = cluster.asks[~cluster.asks.cluster.isna()]

            cluster.ask_iterator = [*range(0, len(cluster.asks))]

            # annotate asking price by grid-fee:
            cluster.asks["adjusted_price"] = cluster.asks["price"] + cluster.asks["cluster"].apply(
                lambda x: self.get_grid_fee(bid_cluster=cluster.idx, ask_cluster=x))

            if len(cluster.bids) == 0:
                # no bids within this cluster: can't match here
                # remove cluster from clusters to match
                cluster.clearing_price_reached = True
                continue

            # order local bids and asks by price
            cluster.bids = cluster.bids.sort_values(["price"], ascending=False)
            cluster.asks = cluster.asks.sort_values(["adjusted_price", "price"],
                                                    ascending=[True, False])

            # match local bids and asks
            cluster.match_locally()

        # Clusters without matches can be discarded
        for cluster in self.clusters:
            if cluster.matched_energy_units <= 0:
                cluster.clearing_price_reached = True

        # Cycle through the clusters.
        # Check the other clusters for the same ask
        # and remove the matches with lower profit. Make sure to change the clearing price
        # each cluster gets an iterator for its index to keep track on checked matches
        for cluster in self.clusters:
            cluster._row = 0

        counter = 0
        while self.clusters_to_match_exist():
            clusters_to_match = self.get_clusters_to_match()
            bid_cluster = clusters_to_match[counter % len(clusters_to_match)]
            idx = bid_cluster._row

            try:
                ask = bid_cluster.asks.iloc[bid_cluster.ask_iterator[idx]]
                assert idx + 1 <= bid_cluster.matched_energy_units
            except (IndexError, AssertionError):
                bid_cluster.clearing_price_reached = True
                continue
            counter += 1

            ask_id = ask.name
            best_match_cluster = self.find_best_profit_cluster(ask_id)
            assert best_match_cluster is not None

            # best cluster to match found. Remove matches from other clusters and adjust their
            # clearing price
            self.remove_from_other_clusters(ask_id, best_match_cluster)

            if best_match_cluster == bid_cluster:
                # if the best match cluster was the current cluster, the row index should increment
                # if not the idx stays the same, since the element was removed.
                bid_cluster._row += 1

        # So far asks, were not removed from dataframes but only from the index in
        # cluster.ask_iterator. This was done since removing single asks is computationally heavy.
        # The dataframes are now set to the remaining indices
        for cluster in self.clusters:
            cluster.asks = cluster.asks.iloc[cluster.ask_iterator]
            cluster.ask_iterator = [*range(0, len(cluster.asks))]

        # All asks for each cluster should be unique now
        # since ask went to the best clusters at a moment when the total matched energy was not
        # decided, the following part runs through all asks, and checks if moving them is profitable
        # for them
        # move ask around / insert them for as long as they find higher profit chances
        asks_changed = True
        # ToDo Might want to have a counter which stops this loop if it does not converge.
        while asks_changed:
            asks_changed = False

            # List tuples (ask, bid_cluster), which has the lowest ask per ask cluster for each
            # (bid) cluster, and is sorted by price, e.g. 2 clusters with bids and asks each, would
            # result in a list of maximum 4 entries with 2 entries for each cluster
            bottom_asks = self.get_bottom_asks()

            for bottom_ask, bid_cluster in bottom_asks:
                best_profit = bid_cluster.clearing_price - bottom_ask.adjusted_price
                if best_profit < 0:
                    continue
                dispute_value = self.resolve_dispute(bottom_ask, bid_cluster)
                clusters = [cluster for cluster in self.clusters if cluster != bid_cluster]
                best_clearing, best_cluster, _ = self.get_best_cluster(dispute_value, best_profit,
                                                                       bottom_ask, clusters)
                if best_cluster is None:
                    # no better cluster found than the current one
                    continue
                asks_changed = True
                bid_cluster.remove(bottom_ask)
                best_cluster.insert(bottom_ask, best_clearing)

        matches = []
        for cluster in self.clusters:
            asks_ = cluster.asks
            bids_ = cluster.bids
            # at this point the matched energy unit should be correct already
            assert cluster.matched_energy_units == \
                   get_clearing(bids_, asks_)["matched_energy_units"]
            for i in range(cluster.matched_energy_units):
                ask = asks_.iloc[i]
                bid = bids_.iloc[i]
                matches.append({
                    "time": self.t_step,
                    "bid_id": bid.name,
                    "ask_id": ask.name,
                    "bid_actor": bid.actor_id,
                    "ask_actor": ask.actor_id,
                    "bid_cluster": bid.cluster,
                    "ask_cluster": ask.cluster,
                    "energy": cfg.config.energy_unit,
                    "price": cluster.clearing_price,
                    "included_grid_fee": ask.adjusted_price - ask.price
                })

        matches = self.group_matches(asks, bids, matches)

        if show:
            print(matches)

        output = self.add_grid_fee_info(matches)
        self.append_to_csv(output, 'matches.csv')

        return matches

    @time_it
    def remove_from_other_clusters(self, ask_id, best_match_cluster):
        for cluster in self.clusters:
            if cluster == best_match_cluster:
                continue
            if cluster.idx is None:
                if ask_id not in cluster.asks.index:
                    # Market Maker asks do not exist, i.e. are not considered
                    # in Market Maker cluster
                    continue

            ask_row = cluster.asks.index.get_loc(ask_id)
            try:
                if cfg.config.debug:
                    print(f"remove row {ask_row} <-> ask_id {ask_id} from cluster {cluster.idx}: "
                          f"{dict(cluster.asks.iloc[ask_row])}")
                cluster.ask_iterator.remove(ask_row)
            except ValueError:
                # ask_id not found. Already deleted
                continue
            cluster.match_locally()

    @time_it
    def get_best_cluster(self, best_dispute_value, best_profit, ask, clusters):
        best_clearing = None
        best_cluster = None
        for cluster in clusters:
            grid_fee = self.get_grid_fee(bid_cluster=cluster.idx, ask_cluster=ask.cluster)
            if cluster.clearing_price - (ask.price + grid_fee) < best_profit:
                # insertion of an ask can only lower the profit. If even the upper bound can not
                # compete with current best profit, skipping the rest increases function speed
                continue
            insertion_profit, clearing = cluster.get_insertion_profit(ask)
            dispute_value = self.resolve_dispute(ask, cluster)
            if (insertion_profit > best_profit or
                    insertion_profit == best_profit and dispute_value > best_dispute_value):
                best_profit = insertion_profit
                best_cluster = cluster
                best_clearing = clearing
                best_dispute_value = dispute_value
        return best_clearing, best_cluster, best_profit

    @time_it
    def get_bottom_asks(self):
        bottom_asks = []
        for cluster in self.clusters:
            if cluster.matched_energy_units == 0:
                continue
            for _cluster in self.clusters:
                asks_with_cluster = cluster.asks[cluster.asks.cluster == _cluster.idx]
                if len(asks_with_cluster) > 0:
                    bottom_asks.append((asks_with_cluster.iloc[0], cluster))
        bottom_asks = sorted(bottom_asks, key=lambda x: x[0].price)
        return bottom_asks

    @time_it
    def find_best_profit_cluster(self, ask_id):
        best_profit = float("-inf")
        best_dispute_value = -float("inf")
        best_match_cluster = None
        # find best cluster for this ask
        # best price AND also capacity to take the energy
        for cluster in self.get_clusters_to_match():
            # Note: ask_price does not influence the best cluster, since the "pure" ask price is
            # the same for all clusters. The all_asks is still used to confirm the ask is
            # not deleted yet
            try:
                ask = cluster.asks.loc[ask_id]
            except KeyError:
                continue
            profit = cluster.clearing_price - ask.adjusted_price
            if cfg.config.debug:
                print(f"? BEST cluster {cluster.idx} profit: {profit} "
                      f"({cluster.clearing_price} - {ask.price})")
            dispute_value = self.resolve_dispute(ask, cluster)
            if ((profit > best_profit and profit >= 0) or
                    (profit == best_profit and dispute_value > best_dispute_value)):
                best_profit = profit
                best_match_cluster = cluster
                best_dispute_value = dispute_value
        if cfg.config.debug:
            if best_match_cluster is not None:
                if best_match_cluster.idx is not None and ask_id not in best_match_cluster.asks:
                    print("Market Maker does not contain Market Maker asks")
                print(f"Found BEST cluster {best_match_cluster.idx} profit: {best_profit} "
                      f"({best_match_cluster.clearing_price} "
                      f"- {best_match_cluster.asks.loc[ask_id].price})")
                cols = ["actor_id", "order_id", "price", "cluster", "adjusted_price"]
                bids = best_match_cluster.bids.reset_index(drop=True)
                asks = best_match_cluster.asks.reset_index(drop=True)
                print(pd.concat([bids[cols[:-2]], asks[cols]], axis=1))

        return best_match_cluster

    @time_it
    def split_orders_to_energy_unit(self, orders):
        orders = pd.DataFrame(orders)
        orders["order_id"] = orders.index
        orders = pd.DataFrame(orders.values.repeat(
            orders.energy * (1 / cfg.config.energy_unit), axis=0), columns=orders.columns)
        orders.energy = cfg.config.energy_unit
        return orders

    @time_it
    def filter_orders(self, asks, bids):
        large_asks_mask = asks.energy >= LARGE_ORDER_THRESHOLD
        large_asks = asks[large_asks_mask]
        asks_mm = large_asks[large_asks.energy >= MARKET_MAKER_THRESHOLD]
        if len(asks_mm) > 1:
            print(f"WARNING! More than one ask market maker:{len(asks_mm)}")
        asks = asks[~large_asks_mask]
        if len(large_asks) > len(asks_mm):
            print("WARNING! {} large asks filtered".format(len(large_asks) - len(asks_mm)))
        large_bids_mask = bids.energy >= LARGE_ORDER_THRESHOLD
        large_bids = bids[large_bids_mask]
        bids_mm = large_bids[large_bids.energy >= MARKET_MAKER_THRESHOLD]
        if len(bids_mm) > 1:
            print(f"WARNING! More than one bid market maker: {len(bids_mm)}")
        bids = bids[~large_bids_mask]
        if len(large_bids) > len(bids_mm):
            print("WARNING! {} large bids filtered".format(len(large_bids) - len(bids_mm)))
        return asks, asks_mm, bids, bids_mm, large_asks, large_bids


@time_it
def get_clearing(bids, asks, prev_clearing_energy: int = None, ask_iterator=None):
    clearing = dict()
    clearing["matched_energy_units"] = 0
    clearing["clearing_price"] = -float("inf")
    clearing["bid_clearing_price"] = None
    start_row = 0
    if prev_clearing_energy:
        start_row = max(prev_clearing_energy - 5, 0)

    if ask_iterator is None:
        ask_iterator = [*range(start_row, len(bids))]
    for row in range(start_row, len(bids)):
        try:
            bid_price = bids.price.iloc[row]
            ask_price = asks.adjusted_price.iloc[ask_iterator[row]]
        except IndexError:
            return clearing
        if ask_price <= bid_price:
            clearing["matched_energy_units"] = row + 1
            clearing["clearing_price"] = ask_price
            clearing["bid_clearing_price"] = bid_price
        else:
            return clearing
    return clearing
