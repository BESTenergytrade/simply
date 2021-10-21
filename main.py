#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import numpy as np

from simply import scenario, market, market_2pac, market_fair
from simply.util import summerize_actor_trading

show_plots = False
show_prints = False

# TODO Config, datetime, etc.
class Config:
    def __init__(self):
        self.start = 8
        self.nb_ts = 3
        self.step_size = 1
        self.list_ts = np.linspace(self.start, self.start + self.nb_ts, self.nb_ts + 1)
        self.path = Path('./scenarios/default')


if __name__ == "__main__":
    cfg = Config()
    sc = scenario.create_random(12, 10)
    # TODO make output folder for config file and Scenario json files, output series in csv and plots files

    if show_plots:
        sc.power_network.plot()
        sc.actors[0].plot(["load", "pv"])

    # Fast forward to interesting start interval for PV energy trading
    for a in sc.actors:
        a.t = cfg.start

    # m = market.Market(0)
    # m = market_2pac.TwoSidedPayAsClear(0)
    m = market_fair.BestMarket(0, sc.power_network)
    for t in cfg.list_ts:
        m.t = t
        for a in sc.actors:
            # TODO concurrent bidding of actors
            order = a.generate_order()
            m.accept_order(order, a.receive_market_results)

        m.clear(reset=True)

        if show_prints:
            print(sc.to_dict())
            m.print()
            print("Matches of bid/ask ids: {}".format(m.matches))
            print(
                "\nCheck individual traded energy blocks (splitted) and price at market level"
            )

    # print("\nTraded energy volume and price at actor level")
    # print(summerize_actor_trading(sc))
