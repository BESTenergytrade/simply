import matplotlib.pyplot as plt

from simply.market import Market
import simply.config as cfg


class TwoSidedPayAsClear(Market):
    """
    Two sided Pay-As-Clear market mechanism, similar to
    https://gridsingularity.github.io/gsy-e/two-sided-pay-as-clear/

    Each timestep, the highest bids are matched with the lowest offers.
    """

    def match(self, show=False):
        # order orders by price
        bids = self.get_bids().sort_values(["price", "energy"], ascending=False)
        asks = self.get_asks().sort_values(["price", "energy"], ascending=True)
        if show:
            plot_merit_order(bids, asks)

        if len(bids) == 0 or len(asks) == 0:
            # no bids or no asks: no match
            return {}

        # match!
        bid_iter = bids.iterrows()
        bid_id, bid = next(bid_iter)
        matches = []
        for ask_id, ask in asks.iterrows():
            while bid is not None:
                if self.grid_fee_matrix:
                    self.apply_grid_fee(ask, bid)
                if ask.price > bid.price:
                    break
                # get common energy value
                energy = min(ask.energy, bid.energy)
                ask.energy -= energy
                bid.energy -= energy
                self.orders.loc[ask_id] = ask
                self.orders.loc[bid_id] = bid
                matches.append({
                    "time": self.t_step,
                    "bid_id": bid_id,
                    "ask_id": ask_id,
                    "bid_actor": bid.actor_id,
                    "ask_actor": ask.actor_id,
                    "bid_cluster": bid.cluster,
                    "ask_cluster": ask.cluster,
                    "energy": energy,
                    "price": ask.price
                })
                if bid.energy + cfg.config.EPS < cfg.config.energy_unit:
                    # bid finished: next bid
                    try:
                        bid_id, bid = next(bid_iter)
                    except StopIteration:
                        bid = None
                if ask.energy + cfg.config.EPS < cfg.config.energy_unit:
                    # ask finished: next ask
                    break

        # adjust price to market clearing price (highest asking price)
        for match in matches:
            match["price"] = matches[-1]["price"]

        if show:
            print(matches)

        output = self.add_grid_fee_info(matches)
        self.append_to_csv(output, 'matches.csv')
        return matches


def plot_merit_order(bids, asks):
    # value asignment in iterrows does not change dataframe -> original shown
    bid_x, bid_y = bids["energy"].to_list(), bids["price"].to_list()
    bid_y = [bid_y[0]] + bid_y
    bid_x_sum = [0] + [sum(bid_x[:(i + 1)]) for i, _ in enumerate(bid_x)]
    ask_x, ask_y = asks["energy"].to_list(), asks["price"].to_list()
    ask_y = [ask_y[0]] + ask_y
    ask_x_sum = [0] + [sum(ask_x[:(i + 1)]) for i, _ in enumerate(ask_x)]

    plt.figure()
    plt.step(bid_x_sum, bid_y, where="pre", label="bids")
    plt.step(ask_x_sum, ask_y, where="pre", label="asks")
    plt.legend()
    plt.xlabel("volume")
    plt.ylabel("price")
    plt.show()
