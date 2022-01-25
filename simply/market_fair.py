import matplotlib.pyplot as plt
import pandas as pd

from simply.market import Market


LARGE_ORDER_THRESHOLD = 2**32
MARKET_MAKER_THRESHOLD = 2**63-1


class BestMarket(Market):
    """
    Custom fair market mechanism.

    Similar to two-sided pay-as-clear, but searches globally for best matches, taking network fees into account.
    Nodes are first grouped into clusters (nodes with no transaction fees between them). Then, all clusters are evaluated individually, adding transaction fees to other clusters. If a match becomes disputed (order matched more than once), the higher offer is taken, while the other one is removed as a possible match and that cluster is re-evaluated. This converges to an optimal solution.
    """

    # clusters is list of sets with node IDs
    clusters = []
    # reverse lookup: node ID -> cluster index
    node_to_cluster = {}
    # matrix with (scaled) weights between clusters
    grid_fee_matrix = []


    def __init__(self, t, network=None, weight_factor=0.1):
        if network is None or not network.network.nodes:
            raise AttributeError("BestMarket needs power network")
        super().__init__(t, network)

        # clustering of nodes by weight. Within cluster, edges have weight 0

        # BFS: start with any node
        nodes = [list(network.network.nodes)[0]]
        while nodes:
            # get first node from list. Guaranteed to not be part of prior cluster
            u = nodes.pop(0)
            # start new cluster with this node
            cluster = len(self.clusters)
            self.clusters.append({u})
            self.node_to_cluster[u] = cluster
            # check neighbors using BFS
            cluster_nodes = [u]
            while cluster_nodes:
                # get next neighbor node
                node = cluster_nodes.pop(0)
                for edge in network.network.edges(node, data = True):
                    # get target of this connection (neighbor of neighbor)
                    v = edge[1]
                    if v in self.node_to_cluster:
                        # already visited
                        continue
                    if edge[2].get("weight", 0) == 0:
                        # weight zero: part of cluster
                        # add to cluster set
                        self.clusters[-1].add(v)
                        self.node_to_cluster[v] = cluster
                        # add to list of neighbors to check later
                        cluster_nodes.append(v)
                    else:
                        # not part of cluster
                        # add to list of nodes that form new clusters
                        nodes.append(v)

        # Calculate accumulated weights on path between clusters and actor nodes
        # Get any one node from each cluster
        root_nodes = {i: list(c)[0] for i, c in enumerate(self.clusters)}
        # init weight matrix with zeros
        num_root_nodes = len(root_nodes)
        self.grid_fee_matrix = [[0]*num_root_nodes for i in range(num_root_nodes)]
        # fill weight matrix
        # matrix symmetric: only need to compute half of values, diagonal is 0
        for i, n1 in root_nodes.items():
            for j, n2 in root_nodes.items():
                if i > j:
                    # get weight between n1 and n2
                    w = self.network.get_path_weight(n1, n2) * weight_factor
                    self.grid_fee_matrix[i][j] = w
                    self.grid_fee_matrix[j][i] = w


    def match(self, show=False):
        asks = self.get_asks()
        bids = self.get_bids()

        # filter out market makers (infinite bus) and really large orders
        large_asks_mask = asks["energy"] >= LARGE_ORDER_THRESHOLD
        large_asks = asks[large_asks_mask]
        asks_mm = large_asks[large_asks["energy"] >= MARKET_MAKER_THRESHOLD]
        asks = asks[~large_asks_mask]
        if len(large_asks) > len(asks_mm):
            print("WARNING! {} large asks filtered".format(len(large_asks) - len(asks_mm)))
        large_bids_mask = bids["energy"] >= LARGE_ORDER_THRESHOLD
        large_bids = bids[large_bids_mask]
        bids_mm = large_bids[large_bids["energy"] >= MARKET_MAKER_THRESHOLD]
        bids = bids[~large_bids_mask]
        if len(large_bids) > len(bids_mm):
            print("WARNING! {} large bids filtered".format(len(large_bids) - len(bids_mm)))

        if (asks.empty and bids.empty) or (asks.empty and asks_mm.empty) or (bids.empty and bid_mm.empty):
            # no asks or bids at all: no matches
            return []

        # split asks and bids into smallest energy unit, save original index, add cluster idx
        asks = pd.DataFrame(asks)
        asks["order_id"] = asks.index
        asks["cluster"] = asks["actor_id"].map(self.node_to_cluster)
        asks = pd.DataFrame(asks.values.repeat(
            asks.energy * (1/self.energy_unit), axis=0), columns=asks.columns)
        asks.energy = self.energy_unit
        bids = pd.DataFrame(bids)
        bids["order_id"] = bids.index
        bids["cluster"] = bids["actor_id"].map(self.node_to_cluster)
        bids = pd.DataFrame(bids.values.repeat(
            bids.energy * (1/self.energy_unit), axis=0), columns=bids.columns)
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
                            "bid_actor": bid.actor_id,
                            "ask_actor": ask.actor_id,
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

        # match with market maker
        # find unmatched orders
        orders = self.orders[(self.orders["energy"] + self.EPS) > self.energy_unit]
        # ignore large orders
        orders = orders[~orders.index.isin(large_asks.index)]
        orders = orders[~orders.index.isin(large_bids.index)]
        # match asks with bid market maker
        asks = orders[orders.type == 1]
        if not bids_mm.empty:
            # bidding market maker: highest price
            bid_mm = bids_mm.sort_values("price", ascending=False).iloc[0]
            asks = asks[asks["price"] <= bid_mm.price]
            matches += list(asks.apply(lambda ask: {
                "time": self.t,
                "bid_actor": bid_mm.actor_id,
                "ask_actor": ask.actor_id,
                "energy": ask.energy,
                "price": bid_mm.price,
            }, axis = 1, result_type = "reduce"))

        # match bids with ask market maker
        bids = orders[orders.type == -1]
        if not asks_mm.empty:
            # asking market maker: lowest price
            ask_mm = asks_mm.sort_values("price", ascending=True).iloc[0]
            bids = bids[bids["price"] >= ask_mm.price]
            matches += list(bids.apply(lambda bid: {
                "time": self.t,
                "bid_actor": bid.actor_id,
                "ask_actor": ask_mm.actor_id,
                "energy": bid.energy,
                "price": ask_mm.price,
            }, axis = 1, result_type = "reduce"))

        if show:
            print(matches)

        return matches
