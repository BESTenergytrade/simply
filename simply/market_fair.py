import pandas as pd

from simply.market import Market

LARGE_ORDER_THRESHOLD = 2**32
MARKET_MAKER_THRESHOLD = 2**63-1


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

    def __init__(self, time, network=None, grid_fee_matrix=None, default_grid_fee=0):
        super().__init__(time, network, grid_fee_matrix, default_grid_fee)

    def match(self, show=False):
        asks = self.get_asks()
        bids = self.get_bids()

        # filter out market makers (infinite bus) and really large orders
        large_asks_mask = asks.energy >= LARGE_ORDER_THRESHOLD
        large_asks = asks[large_asks_mask]
        asks_mm = large_asks[large_asks.energy >= MARKET_MAKER_THRESHOLD]
        asks = asks[~large_asks_mask]
        if len(large_asks) > len(asks_mm):
            print("WARNING! {} large asks filtered".format(len(large_asks) - len(asks_mm)))
        large_bids_mask = bids.energy >= LARGE_ORDER_THRESHOLD
        large_bids = bids[large_bids_mask]
        bids_mm = large_bids[large_bids.energy >= MARKET_MAKER_THRESHOLD]
        bids = bids[~large_bids_mask]
        if len(large_bids) > len(bids_mm):
            print("WARNING! {} large bids filtered".format(len(large_bids) - len(bids_mm)))

        if (asks.empty and bids.empty)\
                or (asks.empty and asks_mm.empty)\
                or (bids.empty and bids_mm.empty):
            # no asks or bids at all: no matches
            return []

        # filter out orders without cluster (can still be matched with market maker)
        asks = asks[~asks.cluster.isna()]
        bids = bids[~bids.cluster.isna()]

        # split asks and bids into smallest energy unit, save original index, add cluster idx
        asks = pd.DataFrame(asks)
        asks["order_id"] = asks.index
        asks = pd.DataFrame(asks.values.repeat(
            asks.energy * (1/self.energy_unit), axis=0), columns=asks.columns)
        asks.energy = self.energy_unit
        bids = pd.DataFrame(bids)
        bids["order_id"] = bids.index
        bids = pd.DataFrame(bids.values.repeat(
            bids.energy * (1/self.energy_unit), axis=0), columns=bids.columns)
        bids.energy = self.energy_unit

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
                            "time": self.t,
                            "bid_id": bid_id,
                            "ask_id": ask_id,
                            "bid_actor": bid.actor_id,
                            "ask_actor": ask.actor_id,
                            "bid_cluster": bid.cluster,
                            "ask_cluster": ask.cluster,
                            "energy": self.energy_unit,
                            "price": ask.adjusted_price,
                            "grid_fee": ask.adjusted_price - ask.price
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

        # group matches: ask -> bid -> match
        _matches = {}
        for match in matches:
            # get original order id of ask/bid and adjust order energy
            ask_id = match["ask_id"]
            ask_order_id = asks.loc[ask_id].order_id
            self.orders.loc[ask_order_id, "energy"] -= match["energy"]
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

        # match with market maker
        # find unmatched orders
        orders = self.orders[(self.orders["energy"] + self.EPS) > self.energy_unit]
        # ignore large orders
        orders = orders[~orders.index.isin(large_asks.index)]
        orders = orders[~orders.index.isin(large_bids.index)]
        # match asks only with bid market maker with highest price
        asks = orders[orders.type == 1]
        if not bids_mm.empty:
            # select bidding market maker by order ID, that has highest price
            bid_mm_id = bids_mm['price'].astype(float).idxmax()
            bid_mm = bids_mm.loc[bid_mm_id]
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
        if not asks_mm.empty:
            # select asking market maker by order ID, that has lowest price
            ask_mm_id = asks_mm['price'].astype(float).idxmin()
            ask_mm = asks_mm.loc[ask_mm_id]
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

        if show:
            print(matches)

        output = self.add_grid_fee_info(matches)
        self.append_to_csv(output, 'matches.csv')

        return matches
