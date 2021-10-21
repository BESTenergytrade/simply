#!/usr/bin/env python3

from argparse import ArgumentParser
from configparser import ConfigParser, MissingSectionHeaderError
import pandas as pd
from pathlib import Path
import numpy as np

from simply import scenario
from simply.market import Market
from simply.util import summerize_actor_trading


# TODO Config, datetime, etc.
class Config:

    def __init__(self, cfg_file):
        parser = ConfigParser()
        try:
            parser.read(cfg_file)
        except MissingSectionHeaderError:
            # headless config: insert missing section header
            with open(cfg_file, "r") as f:
                config_string = '[default]\n' + f.read()
            parser.read_string(config_string)
        self.start = parser.getint("default", "start", fallback = 8)
        self.nb_ts = parser.getint("default", "nb_ts", fallback = 3)
        self.step_size = parser.getint("default", "step_size", fallback = 1)
        self.show_plots = parser.getboolean("default", "show_plots", fallback=False)
        self.show_prints = parser.getboolean("default", "show_prints", fallback=False)
        self.update_scenario = parser.getboolean("default", "update_scenario", fallback=False)
        self.path = parser.get("default", "path", fallback = './scenarios/default')
        self.path = Path(self.path)

        self.list_ts = np.linspace(self.start, self.start + self.nb_ts - 1, self.nb_ts)


if __name__ == "__main__":
    parser = ArgumentParser(description='Entry point for market simulation')
    parser.add_argument('config', nargs='?', default="", help='configuration file')
    args = parser.parse_args()

    cfg = Config(args.config)
    if cfg.path.exists() and not cfg.update_scenario:
        sc = scenario.load(cfg.path)
    else:
        sc = scenario.create_random(12, 11)
        sc.save(cfg.path)
    # TODO make output folder for config file and Scenario json files, output series in csv and plots files

    if cfg.show_plots:
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
        if cfg.show_prints:
            print(sc.to_dict())
            m.print()
            print("Matches of bid/ask ids: {}".format(m.get_all_matches()))
            print(
                "\nCheck individual traded energy blocks (splitted) and price at market level"
            )
            print(m.trades)

    if cfg.show_prints:
        print("\nTraded energy volume and price at actor level")
        print(summerize_actor_trading(sc))
