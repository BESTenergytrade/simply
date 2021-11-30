"""
jh: The script market_2pac orientes itself on the following description:
https://gridsingularity.github.io/d3a/two-sided-pay-as-clear/. It follows the logic of a two-sided-pay-as-cleared-market.
Bids and asks are ordered in a descending and ascending way and are matched as long as the price for asks
is lower then the bidding price.

"""
import matplotlib.pyplot as plt
import pandas as pd

from simply.market import Market


class TwoSidedPayAsClear(Market):
    """
    jh: bids and asks are handed over from the script market.py, sorted and matched. The energy value is handed over.
    The matched energy value is substracted from its initial asking/bidding value till its smaller then 100 kWh
    (100 kWh because its the lowest tradable value).
    """
    def match(self, data, energy_unit=0.1, show=False):
        """
        pay as clear. merit order

        :param data: (Dict[str, Dict]) in format: {"market_name": {{'bids': []], 'offers': []}}
        :param energy_unit: (default: 0.1) minimal energy block in kWh that can be traded
        :param show: (Bool), print final matches
        :return: matches: (Dict) matched orders respectively
        """
        # only a single market is expected
        assert len(data.items()) == 1
        bids = pd.DataFrame(data.get(list(data.keys())[0]).get("bids"))
        asks = pd.DataFrame(data.get(list(data.keys())[0]).get("offers"))
        # keep track of unmatched orders (currently only for debugging purposes)
        orders = pd.concat([bids, asks]).set_index('id')

        # order orders by price
        bids = bids.sort_values(["price", "energy"], ascending=False)
        asks = asks.sort_values(["price", "energy"], ascending=True)

        if len(asks) == 0 or len(bids) == 0:
            # no asks or bids at all: no matches
            return {}

        # match!
        bid_iter = bids.iterrows()
        bid_id, bid = next(bid_iter)
        matches = []
        for ask_id, ask in asks.iterrows():
            while bid is not None and ask.price <= bid.price:
                # get common energy value
                energy = min(ask.energy, bid.energy)
                ask.energy -= energy
                bid.energy -= energy
                # TODO: unmatched orders are not updated/ returned
                orders.loc[ask.id] = ask.drop('id')
                orders.loc[bid.id] = bid.drop('id')
                assert bid.time == ask.time
                matches.append({
                    "time": bid.time,
                    "bid_actor": bid.actor_id,
                    "ask_actor": ask.actor_id,
                    "energy": energy,
                    "price": ask.price
                })
                if ask.energy < energy_unit:
                    # ask finished: next ask
                    break
                if bid.energy < energy_unit:
                    # bid finished: next bid
                    try:
                        bid_id, bid = next(bid_iter)
                    except StopIteration:
                        bid = None

        # adjust price to market clearing price (highest asking price)
        for match in matches:
            match["price"] = matches[-1]["price"]

        if show:
            print(matches)

            # value asignment in iterrows does not change dataframe -> original shown
            bid_x, bid_y = bids["energy"].to_list(), bids["price"].to_list()
            bid_y = [bid_y[0]] + bid_y
            bid_x_sum = [0] + [sum(bid_x[:(i+1)]) for i,_ in enumerate(bid_x)]
            ask_x, ask_y = asks["energy"].to_list(), asks["price"].to_list()
            ask_y = [ask_y[0]] + ask_y
            ask_x_sum = [0] + [sum(ask_x[:(i+1)]) for i,_ in enumerate(ask_x)]

            plt.step(bid_x_sum, bid_y, where="pre", label="bids")
            plt.step(ask_x_sum, ask_y, where="pre", label="asks")
            plt.legend()
            plt.xlabel("volume")
            plt.ylabel("price")
            plt.show()

        return matches
        # return super().match(show)
