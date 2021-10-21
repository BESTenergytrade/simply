import matplotlib.pyplot as plt
import pandas as pd

from simply.market import Market


class BestMarket(Market):

    def __init__(self, t, network=None):
        if network is None or not network.network.nodes:
            raise AttributeError("BestMarket needs power network")
        super().__init__(t, network)

        # clustering of nodes by weight. Within cluster, edges have weight 0
        # clusters is list of sets with node IDs
        self.clusters = []
        # reverse lookup: node ID -> cluster index
        node_to_cluster = {}

        # BFS: start with any node
        nodes = [list(network.network.nodes)[0]]
        while nodes:
            # get first node from list. Guaranteed to not be part of prior cluster
            u = nodes.pop(0)
            # start new cluster with this node
            cluster = len(self.clusters)
            self.clusters.append({u})
            node_to_cluster[u] = cluster
            # check neighbors using BFS
            cluster_nodes = [u]
            while cluster_nodes:
                # get next neighbor node
                node = cluster_nodes.pop(0)
                for edge in network.network.edges(node, data = True):
                    # get target of this connection (neighbor of neighbor)
                    v = edge[1]
                    if v in node_to_cluster:
                        # already visited
                        continue
                    if edge[2].get("weight") == 0:
                        # weight zero: part of cluster
                        # add to cluster set
                        self.clusters[-1].add(v)
                        node_to_cluster[v] = cluster
                        # add to list of neighbors to check later
                        cluster_nodes.append(v)
                    else:
                        # not part of cluster
                        # add to list of nodes that form new clusters
                        nodes.append(v)

    def match(self, show=False):

        asks = self.get_asks()
        bids = self.get_bids()
        if len(asks) == 0 or len(bids) == 0:
            # no asks or bids at all: no matches
            return {}

        # split asks and bids into smallest energy unit, save original index
        asks = pd.DataFrame(asks)
        asks["order_id"] = asks.index
        asks = pd.DataFrame(asks.values.repeat(asks.energy / self.energy_unit, axis=0), columns=asks.columns)
        asks.energy = self.energy_unit
        bids = pd.DataFrame(bids)
        bids["order_id"] = bids.index
        bids = pd.DataFrame(bids.values.repeat(bids.energy / self.energy_unit, axis=0), columns=bids.columns)
        bids.energy = self.energy_unit

        # keep track which clusters have to be (re)matched
        # start with all clusters
        clusters_to_match = set(range(len(self.clusters)))
        # keep track of matches
        matches = []
        # keep track which asks to exclude in each zone (initially empty)
        exclude = {cluster_idx: set() for cluster_idx in clusters_to_match}

        while clusters_to_match:
            # simulate local market within cluster
            cluster_idx = clusters_to_match.pop()
            cluster = self.clusters[cluster_idx]

            # get local bids
            _bids = bids[bids.actor_id.isin(cluster)]
            if len(_bids) == 0:
                # no bids within this cluster: can't match here
                continue

            # get any one node from cluster
            for cluster_root in cluster:
                break

            # get all asks that are not in exclude list for this cluster (copy asks, retain index)
            _asks = asks.drop(exclude[cluster_idx], axis=0, inplace=False)
            # annotate asking price by weight:
            # get node ID for all asks, preserve ordering
            ask_actor_ids = list(_asks.actor_id)
            # get weights from any node in cluster to different ask actors
            weights = self.network.get_cluster_weights([cluster_root], ask_actor_ids)[cluster_root]
            # get weights in same order as ask node IDs
            ask_weights = [weights[i] for i in ask_actor_ids]
            # set adjusted price with network weight
            _asks["adjusted_price"] = pd.Series(_asks.price + ask_weights, index=_asks.index)

            # order local bids and asks by price
            _bids = _bids.sort_values(["price"], ascending=False)
            _asks = _asks.sort_values(["adjusted_price"], ascending=True)

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
                            "bid_actor": int(bid.actor_id),
                            "ask_actor": int(ask.actor_id),
                            "energy": self.energy_unit,
                            "price": ask.adjusted_price,
                            # only for removing doubles later
                            "ask_id": ask_id,
                            "bid_id": bid_id,
                            "cluster": cluster_idx,
                        })
                    # get next bid
                    try:
                        bid_id, bid = next(bid_iter)
                    except (StopIteration, IndexError):
                        bid = None
                # get next ask

            # remove old matches from same cluster
            matches = [m for m in matches if m["cluster"] != cluster_idx]

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
                            exclude[match["cluster"]].add(match["ask_id"])
                            # replace old match
                            matches[match_idx] = _match
                            # redo other cluster
                            clusters_to_match.add(match["cluster"])
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
            # update dictionary
            ask_matches[bid_order_id] = m

        # retrieve matches from nested dict
        matches = [m for ask_matches in _matches.values() for m in ask_matches.values()]

        if show:
            print(matches)

        return matches