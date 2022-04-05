import pandas as pd

from simply.market import Market


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

    def match(self, show=False):
        asks, bids = self.get_asks(), self.get_bids()
        if (asks.empty and bids.empty) \
                or (asks.empty and self.asks_mm.empty) \
                or (bids.empty and self.bids_mm.empty):
            # no asks or bids at all: no matches
            return []

        # filter out orders without cluster (can still be matched with market maker)
        asks = asks[~asks.cluster.isna()]
        bids = bids[~bids.cluster.isna()]

        # split asks and bids into smallest energy unit, save original index, add cluster idx
        asks = pd.DataFrame(asks)
        asks["order_id"] = asks.index
        asks = pd.DataFrame(asks.values.repeat(
            asks.energy * (1 / self.energy_unit), axis=0), columns=asks.columns)
        asks.energy = self.energy_unit
        bids = pd.DataFrame(bids)
        bids["order_id"] = bids.index
        bids = pd.DataFrame(bids.values.repeat(
            bids.energy * (1 / self.energy_unit), axis=0), columns=bids.columns)
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
            # annotate asking price by weight:
            # get cluster ID for all asks, preserve ordering
            ask_cluster_ids = list(_asks.cluster)
            # get weights from any node in cluster to different ask actors
            weights = self.grid_fee_matrix[cluster_idx]
            # get weights in same order as ask node IDs
            ask_weights = [weights[i] for i in ask_cluster_ids]
            # set adjusted price with network weight
            _asks["adjusted_price"] = pd.Series(_asks.price + ask_weights, index=_asks.index)

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
                            "price": ask.adjusted_price
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
                            exclude[match["ask_cluster"]].add(match["ask_id"])
                            # replace old match
                            matches[match_idx] = _match
                            # redo other cluster
                            clusters_to_match.add(match["ask_cluster"])
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

        if show:
            print(matches)

        self.append_to_csv(matches, 'matches.csv')
        return matches
