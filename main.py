#!/usr/bin/env python3

from argparse import ArgumentParser
from simply import scenario, market, market_2pac, market_fair
from simply.actor import Order
from simply.scenario import load, create_random
import os
from simply.config import Config
from simply.util import summerize_actor_trading
from numpy import linspace
from simply.market_fair import MARKET_MAKER_THRESHOLD

"""
Entry point for standalone functionality.

Reads in configuration file (or uses defaults when none is supplied),
creates or loads scenario and matches orders in each timestep.
May show network plots or print market statistics, depending on config.

Usage: python main.py [config file]
"""


if __name__ == "__main__":
    parser = ArgumentParser(description='Entry point for market simulation')
    # parser.add_argument('config', nargs='?', default="", help='configuration file')
    # Replaced the above line to take in the project directory (which will contain the config file) instead of putting in the config file
    # also made it mandatory
    parser.add_argument('project_dir', nargs='?', default=None, help='project directory path')
    args = parser.parse_args()
    # Raise error if project directory not specified
    if args.project_dir is None:
        raise (
            FileNotFoundError("Project directory path must be specified. Please provide the path as a command-line argument."))
    # This means that the config file must always be in the project directory
    config_file = os.path.join(args.project_dir, "config.txt")
    # Raise error if config.txt file not found in project directory
    if not os.path.isfile(config_file):
        raise (
            FileNotFoundError(f"Config file not found in project directory: {args.project_dir}"))
    cfg = Config(config_file, args.project_dir)
    print(cfg)
    # Checks if actor files with the correct format exist in the cfg.scenario_path
    scenario_exists = len([False for i in cfg.scenario_path.glob(f"actor_*.{cfg.data_format}")]) != 0

    print(f'Scenario path: {cfg.scenario_path}')
    # load existing scenario or else create randomized new one
    if cfg.load_scenario:
        if scenario_exists:
            sc = load(cfg.scenario_path, cfg.data_format)
        else:
            raise Exception(f'Could not find scenario path: {cfg.scenario_path}. Make sure to include the '
                            f'scenario directory in your project or if you want to generate a random scenario, '
                            f'set load_scenario = False in config.txt.')
    else:
        sc = create_random(cfg.nb_nodes, cfg.nb_actors, cfg.weight_factor)
        sc.save(cfg.scenario_path, cfg.data_format)

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

    # Actors generate their order prices based on the prices communicated by the market maker.
    # These need to be the same for each actor. This is guaranteed by the loop below.
    # ToDo: Store market maker in a single data structure, e.g. market or scenario object
    for a in sc.actors:
        a.data.selling_price = sc.actors[0].data.selling_price
        a.data.price = sc.actors[0].data.price
        a.create_prediction()

    for t in list_ts:
        m.t = t
        for a in sc.actors:
            # actor calculates strategy based market interaction with the market maker
            a.get_market_schedule()
            # orders are generated based on the flexibility towards the planned market interaction
            # and a pricing scheme
            order = a.generate_order()
            if order:
                m.accept_order(order, callback=a.receive_market_results)

        # Generate market maker order as ask
        m.accept_order(
            Order(1, t, 'market_maker', None, MARKET_MAKER_THRESHOLD,
                  sc.actors[0].pred.price[0]))
        # Generate market maker order as bid
        m.accept_order(
            Order(-1, t, 'market_maker', None, MARKET_MAKER_THRESHOLD,
                  sc.actors[0].pred.selling_price[0]))

        m.clear(reset=cfg.reset_market)
        for a in sc.actors:
            # Update all actors for the next market time slot
            a.next_time_step()

        if cfg.show_prints:
            print("Matches of bid/ask ids: {}".format(m.matches))
            print(
                "\nCheck individual traded energy blocks (splitted) and price at market level"
            )

    if cfg.show_prints:
        print("\nTraded energy volume and price at actor level")
        print(summerize_actor_trading(sc))
