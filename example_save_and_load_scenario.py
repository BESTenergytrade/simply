#!/usr/bin/env python3

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
    cfg = Config()
    sc = scenario.create_random(12, 10)
    # TODO make output folder for config file and Scenario json files, output series in csv and
    #  plots files

    sc.save(cfg.path)

    del sc
    sc = scenario.load(cfg.path)
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

    print("\nCheck traded energy volume and price at actor level")
    print(summerize_actor_trading(sc))
