#!/usr/bin/env python3

from argparse import ArgumentParser

from simply import scenario, market, market_2pac, market_fair
from simply.config import Config
from simply.util import summerize_actor_trading
from numpy import linspace


"""
Entry point for standalone functionality.

Reads in configuration file (or uses defaults when none is supplied),
creates or loads scenario and matches orders in each timestep.
May show network plots or print market statistics, depending on config.

Usage: python main.py [config file]
"""


if __name__ == "__main__":
    parser = ArgumentParser(description='Entry point for market simulation')
    parser.add_argument('config', nargs='?', default="", help='configuration file')
    args = parser.parse_args()

    cfg = Config(args.config)
    # Load scenario, if path exists with the correct format
    # otherwise remove all files in existing folder and create new scenario
    # check if actor-files with correct format exist in cfg.path
    scenario_exists = len([False for i in cfg.path.glob(f"actor_*.{cfg.data_format}")]) != 0

    print(f'Scenario path: {cfg.path}')
    # load existing scenario or else create randomized new one
    if cfg.load_scenario:
        if scenario_exists:
            sc = scenario.load(cfg.path, cfg.data_format)
        else:
            raise Exception(f'Could not find actor data in path: {cfg.path} .')
    else:
        if cfg.path.exists():
            raise Exception(f'The path: {cfg.path} already exists with another file structure. '
                            'Please remove or rename folder to avoid confusion and restart '
                            'simulation.')
        sc = scenario.create_random(cfg.nb_nodes, cfg.nb_actors, cfg.weight_factor)
        sc.save(cfg.path, cfg.data_format)

    if cfg.show_plots:
        sc.power_network.plot()
        sc.plot_actor_data()

    # Fast forward to interesting start interval for PV energy trading
    for a in sc.actors:
        a.t = cfg.start

    # generate requested market
    if "pac" in cfg.market_type:
        m = market_2pac.TwoSidedPayAsClear(0, network=sc.power_network)
    elif cfg.market_type in ["fair", "merit"]:
        m = market_fair.BestMarket(0, sc.power_network)
    else:
        # default
        m = market.Market(0, network=sc.power_network)

    list_ts = linspace(cfg.start, cfg.start + cfg.nb_ts - 1, cfg.nb_ts)
    for t in list_ts:
        m.t = t
        for a in sc.actors:
            order = a.generate_order()
            if order:
                m.accept_order(order, callback=a.receive_market_results)

        m.clear(reset=cfg.reset_market)
        if cfg.show_prints:
            print("Matches of bid/ask ids: {}".format(m.matches))
            print(
                "\nCheck individual traded energy blocks (splitted) and price at market level"
            )

    if cfg.show_prints:
        print("\nTraded energy volume and price at actor level")
        print(summerize_actor_trading(sc))
