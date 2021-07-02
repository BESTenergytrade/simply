import matplotlib.pyplot as plt

from simply.market import Market


class TwoSidedPayAsClear(Market):

    def match(self, show=False):
        orders = self.get_order_df()
        # order orders by price
        bids = orders[orders["type"] > 0].sort_values(["price", "energy"], ascending=False)
        asks = orders[orders["type"] < 0].sort_values(["price", "energy"])

        if len(bids) == 0 or len(asks) == 0:
            # no bids or no asks: no match
            return {}

        # match!
        bid_iter = bids.iterrows()
        bid = next(bid_iter)[1]
        matches = []
        for _, ask in asks.iterrows():
            while bid is not None and ask.price <= bid.price:
                # get common energy value
                energy = min(ask.energy, bid.energy)
                ask.energy -= energy
                bid.energy -= energy
                matches.append({
                    "bid_actor": int(bid.actor_id),
                    "ask_actor": int(ask.actor_id),
                    "energy": energy,
                    "price": ask.price
                })
                if ask.energy == 0:
                    # ask finished: next ask
                    break
                if bid.energy == 0:
                    # bid finished: next bid
                    try:
                        bid = next(bid_iter)[1]
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
