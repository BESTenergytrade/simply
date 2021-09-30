"""JuH: the script "main.py" inserts the Config class which gives a starting point and intervall for the simulation. The scenario.py script is loaded in which already includes the actor.py and power_network.py.
In the first for loop every actor with it's features is loaded. In the second loop each actor passes through each time step and lines up each bids and asks. The market is then cleared."""


#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import numpy as np

from simply import scenario
from simply.market import Market
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
    """ JuH: bids and asks are loaded into the script and orders are generated. Sums up the bids, asks, actor_id and the price per timestamp from 8-11 (after each loop)"""
    cfg = Config()
    sc = scenario.create_random(12, 10)
    # TODO make output folder for config file and Scenario json files, output series in csv and plots files

    if show_plots:
        sc.actors[0].plot(["load", "pv"])

    # Fast forward to interesting start interval for PV energy trading
    for a in sc.actors:
        a.t = cfg.start

    for t in cfg.list_ts:
        m = Market(t)
        for a in sc.actors:
            # TODO concurrent bidding of actors
            order = a.generate_order()
            m.accept_order(order, a.receive_market_results)

        m.clear()
        if show_prints:
            print(sc.to_dict())
            m.print()
            print("Matches of bid/ask ids: {}".format(m.get_all_matches()))
            print(
                "\nCheck individual traded energy blocks (splitted) and price at market level"
            )
            print(m.trades)

    print("\nTraded energy volume and price at actor level")
    print(summerize_actor_trading(sc))
