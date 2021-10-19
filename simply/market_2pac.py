"""
The script market_2pac orientes itself on the following description:
https://gridsingularity.github.io/d3a/two-sided-pay-as-clear/. It follows the logic of a two-sided-pay-as-cleared-market.
Bids and asks are ordered in a descending and ascending way and are matched as long as the price for asks
is lower then the bidding price.

"""



import matplotlib.pyplot as plt

from simply.market import Market


class TwoSidedPayAsClear(Market):
"""bids and asks are handed over from the script market.py, sorted and matched. The energy value is handed over.
The matched energy value is substracted from its initial asking/bidding value till its smaller then 100 kWh
(100 kWh because its the lowest tradable value)."""
    def match(self, show=True):
        # order orders by price
        bids = self.get_bids().sort_values(["price", "energy"], ascending=False)
        asks = self.get_asks().sort_values(["price", "energy"], ascending=True)

        if len(bids) == 0 or len(asks) == 0:
            # no bids or no asks: no match
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
                self.orders.loc[ask_id] = ask
                self.orders.loc[bid_id] = bid
                matches.append({
                    "time": self.t,
                    "bid_actor": int(bid.actor_id),
                    "ask_actor": int(ask.actor_id),
                    "energy": energy,
                    "price": ask.price
                })
                if ask.energy < self.energy_unit:
                    # ask finished: next ask
                    break
                if bid.energy < self.energy_unit:
                    # bid finished: next bid
                    try:
                        bid_id, bid = next(bid_iter)
                    except IndexError:
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
