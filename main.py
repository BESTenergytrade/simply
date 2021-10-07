#!/usr/bin/env python3

import argparse
import json
import pandas as pd
from pathlib import Path
import numpy as np

from simply import scenario, market, market_2pac, market_fair
from simply.util import summerize_actor_trading

# TODO Config, datetime, etc.
class Config:

    start = 8
    nb_ts = 3
    step_size = 1
    path = './scenarios/default'
    update_scenario = False
    show_plots = False
    show_prints = False

    def __init__(self, filepath):
        # read config from filepath (if given)
        if filepath is not None:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#'):
                        # comment
                        continue
                    if len(line) == 0:
                        # empty line
                        continue
                    k, v = line.split('=')
                    k = k.strip()
                    v = v.strip()
                    try:
                        # option may be special: number, array, etc.
                        v = json.loads(v)
                    except ValueError:
                        # or not
                        pass
                    #set config option
                    vars(self)[k] = v
        # end read config
        self.path = Path(self.path)
        self.list_ts = np.linspace(self.start, self.start + self.nb_ts - 1, self.nb_ts)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Entry point for market simulation')
    parser.add_argument('config', nargs='?', help='configuration file')
    args = parser.parse_args()

    cfg = Config(args.config)

    if cfg.path.exists() and not cfg.update_scenario:
        sc = scenario.load(cfg.path)
    else:
        sc = scenario.create_random(12, 11)
        sc.save(cfg.path)

    # TODO make output folder for config file, output series (csv, plot) files

    if cfg.show_plots:
        sc.power_network.plot()
        sc.actors[0].plot(["load", "pv"])

    # Fast forward to interesting start interval for PV energy trading
    for a in sc.actors:
        a.t = cfg.start

    asks_list = []
    bids_list = []

    # m = market.Market(0)
    # m = market_2pac.TwoSidedPayAsClear(0)
    m = market_fair.BestMarket(0, sc.power_network)
    for t in cfg.list_ts:
        m.t = t
        for a in sc.actors:
            # TODO concurrent bidding of actors
            order = a.generate_order()
            m.accept_order(order, a.receive_market_results)

        asks_list.append(m.get_asks())
        bids_list.append(m.get_bids())

        m.clear(reset=True)

        if cfg.show_prints:
            print(sc.to_dict())
            # TODO depricated:
            # m.print()
            print("Matches of bid/ask ids: {}".format(m.matches))
            print(
                "\nCheck individual traded energy blocks (splitted) and price at market level"
            )

    if cfg.show_prints:
        print("\nEnergy bids and asks")
        print(asks_list)
        print(bids_list)
        print("\nTraded energy volume and price at actor level")
        print(summerize_actor_trading(sc))
        sc.power_network.plot()
