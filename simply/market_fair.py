import pandas as pd

from simply.market import Market
import simply.config as cfg

from simply.market import LARGE_ORDER_THRESHOLD
from simply.market import MARKET_MAKER_THRESHOLD


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

    def __init__(self, network=None, grid_fee_matrix=None, time_step=None, disputed_matching='price',):
        super().__init__(network, grid_fee_matrix, time_step)
        self.disputed_matching = disputed_matching

    def match(self, show=False):
        asks = self.get_asks()
        bids = self.get_bids()

        # filter out market makers (infinite bus) and really large orders
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
            asks.energy * (1/cfg.config.energy_unit), axis=0), columns=asks.columns)
        asks.energy = cfg.config.energy_unit
        bids = pd.DataFrame(bids)
        bids["order_id"] = bids.index
        bids = pd.DataFrame(bids.values.repeat(
            bids.energy * (1/cfg.config.energy_unit), axis=0), columns=bids.columns)
        bids.energy = cfg.config.energy_unit

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


    def match_single_loop(self, show=False):
        asks = self.get_asks()
        bids = self.get_bids()

        # filter out market makers (infinite bus) and really large orders
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

        if (asks.empty and bids.empty) \
                or (asks.empty and asks_mm.empty) \
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
            asks.energy * (1 / cfg.config.energy_unit), axis=0), columns=asks.columns)
        asks.energy = cfg.config.energy_unit
        bids = pd.DataFrame(bids)
        bids["order_id"] = bids.index
        bids = pd.DataFrame(bids.values.repeat(
            bids.energy * (1 / cfg.config.energy_unit), axis=0), columns=bids.columns)
        bids.energy = cfg.config.energy_unit

        # keep track which clusters have to be (re)matched
        # start with all clusters
        clusters_to_match = set(range(len(self.grid_fee_matrix)))

        # keep track of matches, bids and asks for each bid cluster separately
        all_matches = {cluster_idx: [] for cluster_idx in clusters_to_match}

        # this will store the sorted Dataframe for each cluster later on
        matched_bids = {cluster_idx: pd.DataFrame() for cluster_idx in clusters_to_match}

        all_asks = {cluster_idx: pd.DataFrame() for cluster_idx in clusters_to_match}
        all_bids = {cluster_idx: pd.DataFrame() for cluster_idx in clusters_to_match}

        all_clearing_prices = {cluster_idx: float('-inf') for cluster_idx in clusters_to_match}
        # exclusion list is not needed anymore since matching is not done iteratively anymore

        # keep track which asks to exclude in each zone (initially empty)
        # exclude = {cluster_idx: set() for cluster_idx in clusters_to_match}

        # iterate over all clusters and match the cluster bids with all the asks. This will lead to
        # asks full filling multiple bids. These duplicate matches will be taken care of later
        clusters_to_match_copy = {idx for idx in clusters_to_match}

        # Work with copy to be able to remove empty bid clusters
        for cluster_idx in clusters_to_match_copy:
            # simulate local market within cluster

            # initialize current matches
            matches = all_matches[cluster_idx]

            # get local bids
            _bids = bids[bids.cluster == cluster_idx]
            if len(_bids) == 0:
                # no bids within this cluster: can't match here
                # remove cluster from clusters to match
                clusters_to_match.remove(cluster_idx)
                continue

            _asks = asks.copy()

            # annotate asking price by grid-fee:
            # get cluster ID for all asks, preserve ordering
            ask_cluster_ids = list(_asks.cluster)
            # get grid_fees from any node in cluster to different ask actors
            # get grid_fees in same order as ask node IDs

            ask_grid_fees = [self.get_grid_fee(bid_cluster=cluster_idx,ask_cluster=i)
                             for i in ask_cluster_ids]
            # set adjusted price with network grid-fee
            _asks["adjusted_price"] = pd.Series(_asks.price + ask_grid_fees, index=_asks.index)

            # order local bids and asks by price
            _bids = _bids.sort_values(["price"], ascending=False)
            _asks = _asks.sort_values(["adjusted_price", "price"], ascending=[True, False])

            # Store the sorted and adjusted asks
            all_asks[cluster_idx] = _asks.copy()
            all_bids[cluster_idx] = _bids.copy()

            # match local bids and asks
            bid_iter = _bids.iterrows()
            bid_id, bid = next(bid_iter)
            _matches = []
            highest_bid_row = -1
            for ask_id, ask in _asks.iterrows():
                # compare with next bid
                if bid is not None:
                    # still bids in queue
                    if ask.adjusted_price <= bid.price:
                        # the lowest bid which still got matched. Taking the row is equivalent to
                        # the energy amount.
                        highest_bid_row = _bids.index.get_loc(bid.name)
                    else:
                        # bid price is lower than ask. Continuing iteration does not make sense
                        break

                    # get next bid
                    try:
                        bid_id, bid = next(bid_iter)
                    except (StopIteration, IndexError):
                        break
                # get next ask

            matched_bids[cluster_idx] = _bids.iloc[0:highest_bid_row+1].copy()


        # All bids are matched with all asks. Get the clearing price for each cluster
        for cluster_idx in clusters_to_match.copy():
            if len(matched_bids[cluster_idx]) == 0:
                # No matches in this bid cluster although all asks were available. No need to check
                # this cluster further
                clusters_to_match.remove(cluster_idx)
            else:
                amount = len(matched_bids[cluster_idx])
                # amount of 1 should lead to clearing price of row=0 --> decrement amount by 1
                row = amount-1
                all_clearing_prices[cluster_idx] = all_asks[cluster_idx].iloc[row]["adjusted_price"]

        # Cycle through the clusters.
        # Check the other clusters for the same ask
        # and remove the matches with lower profit. Make sure to change the clearing price
        # each cluster gets an iterator for its index to keep track on checked matches
        index_iterators = {idx: iter(range(len(matched_bids[idx]))) for idx in clusters_to_match}
        indices = {idx: next(index_iterators[idx]) for idx in clusters_to_match}
        clusters_to_match_copy = {idx for idx in clusters_to_match}

        counter = 0
        while len(clusters_to_match_copy)>0:
            clusters_to_match = {idx for idx in clusters_to_match_copy}

            current_cluster = list(clusters_to_match)[counter % len(clusters_to_match)]
            idx = indices[current_cluster]
            counter += 1

            cluster_idx = current_cluster
            try:
                ask = all_asks[cluster_idx].iloc[idx]
            except IndexError:
                clusters_to_match_copy.remove(cluster_idx)
                continue

            ask_id = ask.name
            best_profit = float("-inf")
            best_match_cluster = current_cluster

            # find best cluster for this ask
            # best price AND also capacity to take the energy
            for ii in clusters_to_match:
                # Note: ask_price does not influence the best cluster, since the "pure" ask price is
                # the same for all clusters. The all_asks is still used to confirm the ask is
                # not deleted yet
                try:
                    profit = all_clearing_prices[ii] - all_asks[ii].loc[ask_id]["adjusted_price"]
                except KeyError:
                    continue

                energy_in_bids_to_match = len(matched_bids[ii]) - 1
                energy_in_asks = all_asks[ii].index.get_loc(ask_id)
                if profit > best_profit and energy_in_bids_to_match>=energy_in_asks:
                    best_profit = profit
                    best_match_cluster = ii

            # best cluster to match found. Remove matches from other clusters and adjust their
            # clearing price
            for ii in clusters_to_match:
                if ii == best_match_cluster:
                    continue
                # drop the ask from the asks in the "bad" cluster
                try:
                    all_asks[ii]=all_asks[ii].drop(ask_id)
                except KeyError:
                    # ask_id not found. Already deleted
                    continue

                # update clearing price and make sure the matched_bids get adjusted if need be
                bid_price = matched_bids[ii].iloc[-1]["price"]
                row = len(matched_bids[ii]) - 1
                if (all_asks[ii].iloc[row]["adjusted_price"] == all_clearing_prices[ii] and
                    all_asks[ii].iloc[row]["adjusted_price"] < bid_price):
                    # all good, ask was removed, but another ask can still fulfill the same energy
                    # amount for the same clearing price
                    pass
                elif all_asks[ii].iloc[row]["adjusted_price"] <= bid_price:
                    # same amount of bids are met, but for a higher clearing price.
                    all_clearing_prices[ii] = all_asks[ii].iloc[row]["adjusted_price"]
                else:
                    # matched energy for the cluster has to be reduced. clearing price remains
                    # or in case the highest ask was removed it sinks
                    assert all_clearing_prices[ii] >= all_asks[ii].iloc[row-1]["adjusted_price"]
                    matched_bids[ii].drop(matched_bids[ii].iloc[-1].name, inplace=True)
                    row = len(matched_bids[ii]) - 1
                    all_clearing_prices[ii] = all_asks[ii].iloc[row]["adjusted_price"]

            if best_match_cluster == cluster_idx:
                # if the best match cluster was the current cluster, the index should increment
                # if not the idx stays the same, since the element was removed.
                try:
                    indices[cluster_idx] = next(index_iterators[cluster_idx])
                except StopIteration:
                    clusters_to_match_copy.remove(cluster_idx)

        # All asks for each cluster should be unique now
        # assert len(set(all_asks[0].index[0:len(matched_bids[0])]).intersection(all_asks[1].index[0:len(matched_bids[1])])) == 0

        # since ask went to the best clusters at a moment when the total matched energy was not
        # decided, the following part runs through all asks, and checks if moving them is profitable
        # for them

        # move ask around / insert them for as long as they find higher profit chances
        asks_changed = True
        counter=0
        while asks_changed:
            # leave loop if nothing changes. Set asks_changed to true if smth changed this
            # iteration
            counter +=1
            if counter>100:
                warnings.warn("Balancing markets through profit oriented shuffling does not"
                              "converge", stacklevel=100)
                break
            asks_changed = False

            bottom_asks=[]
            clusters = [i for i in matched_bids]
            for i, df_bids in matched_bids.items():
                if len(df_bids) == 0:
                    continue
                for cluster_idx in clusters:
                    asks_with_cluster= all_asks[i][all_asks[i].cluster == cluster_idx]
                    if len(asks_with_cluster)>0:
                        bottom_asks.append((asks_with_cluster.iloc[0], i))

                # bottom_asks.append((all_asks[i].iloc[0], i))

            bottom_asks = sorted(bottom_asks, key=lambda x: x[0].price)
            for bottom_ask,i in bottom_asks:
                best_profit = float("-inf")
                best_insert_cluster = None
                best_lower_bid = None
                best_clearing_price = float("-inf")
                best_insertion_price = float("-inf")
                for ii, df_bids_2 in matched_bids.items():
                    # check how the ask would interact in another cluster ii
                    if i == ii:
                        continue
                    # what would be the price of the ask in this other cluster be
                    insertion_price = bottom_ask.price+self.grid_fee_matrix[bottom_ask.cluster][ii]
                    if insertion_price >= all_clearing_prices[ii]:
                        # if its worse than the clearing price, it will not be inserted
                        continue
                    # ask is lower than clearing price of other cluster. Insertion only makes sense
                    # if the ask does not become the new clearing price

                    next_lower_bid = all_bids[ii].iloc[len(df_bids_2)]
                    if next_lower_bid.price < all_clearing_prices[ii]:
                        # inserting an ask will move the highest ask out of the matches
                        # clearing price will be the second highest ask before insertion OR the new ask
                        new_clearing_price = max(all_asks[ii].iloc[len(df_bids_2)-2].adjusted_price,
                                                 insertion_price)
                    else:
                        # next bid is still over clearing price. Insertion will not change the
                        # clearing price, but the matched bids need to bee appended
                        new_clearing_price = all_clearing_prices[ii]

                    profit_new = new_clearing_price-insertion_price
                    if profit_new > best_profit:
                        best_profit = profit_new
                        best_clearing_price = new_clearing_price
                        best_insertion_price= insertion_price
                        best_insert_cluster = ii
                        best_lower_bid = next_lower_bid

                insertion_price = best_insertion_price
                new_clearing_price = best_clearing_price
                next_lower_bid = best_lower_bid
                ii = best_insert_cluster
                profit_new = best_profit

                profit_before = all_clearing_prices[i]-bottom_ask.adjusted_price
                if profit_new > profit_before:
                    print(f"Moving Actor {bottom_ask.actor_id} from cluster {i} with clearing price"
                          f" of {all_clearing_prices[i]} to cluster{best_insert_cluster} with new clearing price of"
                          f" {best_clearing_price}")

                    # ask will be moved from cluster i to other cluster ii
                    asks_changed = True
                    all_asks[i] = all_asks[i].drop(bottom_ask.name)
                    new_clearing_price_i=all_asks[i].iloc[len(matched_bids[i])-1].adjusted_price

                    original_clearing_bid = all_bids[i].iloc[len(matched_bids[i])-1]

                    if new_clearing_price_i <= matched_bids[i].iloc[-1].price:
                        # all good. New clearing price of more expensive ask is below bid cap
                        all_clearing_prices[i] = new_clearing_price_i

                    else:
                        # removing the ask reduced the matched energy. Adjust clearing price
                        # and matched bids
                        new_clearing_price_i = all_asks[i].iloc[len(matched_bids[i])-2].adjusted_price
                        all_clearing_prices[i] = new_clearing_price_i
                        matched_bids[i]=matched_bids[i].drop(matched_bids[i].iloc[-1].name)


                    all_asks[ii].loc[bottom_ask.name] = bottom_ask
                    all_asks[ii].loc[bottom_ask.name, "adjusted_price"] = insertion_price
                    all_asks[ii] = all_asks[ii].sort_values(by=["adjusted_price", "price"])
                    all_clearing_prices[ii] = new_clearing_price

                    new_cap_ask_ii=all_asks[ii].iloc[len(matched_bids[ii])-1]
                    # new cap of ii could also be the cap of i. Therefore its moved and the original
                    # clearing price for ii is recovered

                    # original_clearing_bid_price_i=original_clearing_bid.price
                    # if (new_cap_ask_ii.price + self.grid_fee_matrix[new_cap_ask_ii.cluster][ii]
                    #         <= original_clearing_bid_price_i):
                    #     all_asks[ii]=all_asks[ii].drop(new_cap_ask_ii.name)
                    #     all_clearing_prices[ii] = all_asks[ii].iloc[len(matched_bids[ii])-1].adjusted_price
                    #
                    #     all_asks[i].loc[new_cap_ask_ii.name] = new_cap_ask_ii
                    #     new_cap_adjusted_price = new_cap_ask_ii.price + \
                    #                              self.grid_fee_matrix[new_cap_ask_ii.cluster][i]
                    #     all_clearing_prices[i] = new_cap_adjusted_price
                    #     all_asks[i].loc[new_cap_ask_ii.name, "adjusted_price"] = new_cap_adjusted_price
                    #     all_asks[i] = all_asks[i].sort_values(by=["adjusted_price", "price"])
                    #     matched_bids[i].loc[original_clearing_bid.name] = original_clearing_bid



                    appended_clearing_price=all_asks[ii].iloc[len(matched_bids[ii])].adjusted_price
                    if next_lower_bid.price >= appended_clearing_price:
                        matched_bids[ii].loc[next_lower_bid.name] = next_lower_bid
                        all_clearing_prices[ii] = appended_clearing_price




                    # cluster i lost an ask. An unmatched ask might want to cap this cluster
                    next_bid = all_bids[i].iloc[len(matched_bids[i])]
                    lowest_ask_price = float("inf")
                    current_cluster = None
                    for iii, df_bids_iii in matched_bids.items():
                        if len(df_bids_iii) == 0:
                            continue
                        # is unmatched ask viable for capping cluster i?
                        checked_ask = all_asks[iii].iloc[len(df_bids_iii)]
                        if checked_ask.price+self.grid_fee_matrix[checked_ask.cluster][i] < lowest_ask_price:
                            ask_4_cap = checked_ask
                            lowest_ask_price = ask_4_cap.price+self.grid_fee_matrix[ask_4_cap.cluster][i]
                            current_cluster = iii
                    if lowest_ask_price < next_bid.price:
                        # an ask was found to fill the gap left by moving the ask.
                        # adjust clearing price of cluster, drop ask from old cluster,
                        # add ask to new cluster and append matched bids

                        all_asks[current_cluster] = all_asks[current_cluster].drop(ask_4_cap.name)
                        all_asks[i].loc[ask_4_cap.name] = ask_4_cap
                        all_asks[i].loc[ask_4_cap.name, "adjusted_price"] = ask_4_cap.price+self.grid_fee_matrix[ask_4_cap.cluster][i]
                        all_asks[i] = all_asks[i].sort_values(by=["adjusted_price", "price"])
                        matched_bids[i].loc[next_bid.name]=next_bid
                        all_asks[i] = all_asks[i].sort_values(by=["price"])
                        # sometimes, an inserted ask pushes the previous last match out of
                        # matching. Here it is checked that no impossible matches exist
                        impossible_matches = True
                        while impossible_matches:
                            impossible_matches = False
                            row = matched_bids[i].index.get_loc(matched_bids[i].iloc[-1].name)
                            if all_asks[i].iloc[row].adjusted_price > matched_bids[i].iloc[row].price:
                                matched_bids[i]=matched_bids[i].drop(matched_bids[i].iloc[row].name)
                                impossible_matches = True
                        all_clearing_prices[i] = all_asks[i].iloc[row].adjusted_price

                        print(
                            f"Capping Actor {ask_4_cap.actor_id} from cluster {current_cluster} with clearing price"
                            f" of {all_clearing_prices[current_cluster]} to cluster{i} with new clearing price of"
                            f" {all_clearing_prices[i]}")

                    break

                else:
                    # ask would become the new clearing price. since the profit is zero it
                    # would only make sense, in case the ask was not matched before.
                    # Since only matched bids are checked right now, no need for checking
                    # if it was matched before is needed
                    continue


        for i, _ in enumerate(matched_bids.values()):
            asks = all_asks[i]
            m_bids = matched_bids[i]
            for ii, single_bid in enumerate(m_bids.itertuples()):
                matches.append({
                    "time": self.t_step,
                    "bid_id": single_bid.Index,
                    "ask_id": asks.iloc[ii].name,
                    "bid_actor": single_bid.actor_id,
                    "ask_actor": asks.iloc[ii].actor_id,
                    "bid_cluster": single_bid.cluster,
                    "ask_cluster": asks.iloc[ii].cluster,
                    "energy": cfg.config.energy_unit,
                    "price": all_clearing_prices[i],
                    "included_grid_fee": asks.iloc[ii].adjusted_price - asks.iloc[ii].price
                })
        return matches


    def remove_double_matches(self, exclude, match, _match, clusters_to_match, matches, bids,
                              match_idx=None):
        if self.disputed_matching == 'price':
            if _match["price"] > match["price"]:
                # new match is better:
                # exclude old match
                self.exclude_matches(exclude, match, _match, clusters_to_match, matches,
                                     match_idx=match_idx, exclude_exisiting=True)
            else:
                # old match is better: exclude new match
                self.exclude_matches(exclude, _match, match, clusters_to_match, matches,
                                     exclude_exisiting=False)

        elif self.disputed_matching == 'grid_fee':
            if _match["included_grid_fee"] < match["included_grid_fee"] \
                    or _match["included_grid_fee"] == match["included_grid_fee"] \
                    and bids.loc[_match['bid_id'], 'price'] > bids.loc[match['bid_id'], 'price']:
                # new match is better:
                # exclude old match
                self.exclude_matches(exclude, match, _match, clusters_to_match,
                                     matches, match_idx=match_idx, exclude_exisiting=True)
            else:
                # old match is better: exclude new match
                self.exclude_matches(exclude, _match, match, clusters_to_match, matches,
                                     exclude_exisiting=False)

        elif self.disputed_matching == 'bid_price':
            if bids.loc[_match['bid_id'], 'price'] > bids.loc[match['bid_id'], 'price'] or \
                    bids.loc[_match['bid_id'], 'price'] == bids.loc[match['bid_id'], 'price'] and \
                    _match['included_grid_fee'] < match['included_grid_fee']:
                # new match is better:
                # exclude old match
                self.exclude_matches(exclude, match, _match, clusters_to_match, matches,
                                     match_idx=match_idx, exclude_exisiting=True)
            else:
                # old match is better: exclude new match
                self.exclude_matches(exclude, _match, match, clusters_to_match, matches,
                                     exclude_exisiting=False)
        elif self.disputed_matching == 'profit':
            if _match["price"] - _match["included_grid_fee"] > \
                    match["price"] - match["included_grid_fee"]:
                # new match is better:
                # exclude old match
                self.exclude_matches(exclude, match, _match, clusters_to_match, matches,
                                     match_idx=match_idx, exclude_exisiting=True)
            else:
                # old match is better: exclude new match
                self.exclude_matches(exclude, _match, match, clusters_to_match, matches,
                                     exclude_exisiting=False)
