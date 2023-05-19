#!/usr/bin/env python3

from argparse import ArgumentParser

from simply import market, market_2pac, market_fair
from simply.actor import Order
from simply.scenario import load, create_random
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
            sc = load(cfg.path, cfg.data_format)
        else:
            raise Exception(f'Could not find actor data in path: {cfg.path} .')
    else:
        if cfg.path.exists():
            raise Exception(f'The path: {cfg.path} already exists with another file structure. '
                            'Please remove or rename folder to avoid confusion and restart '
                            'simulation.')
        sc = create_random(cfg.nb_nodes, cfg.nb_actors, cfg.weight_factor)
        sc.save(cfg.path, cfg.data_format)

    if cfg.show_plots:
        sc.power_network.plot()
        sc.plot_actor_data()

    # Fast forward to interesting start interval for PV energy trading
    for a in sc.actors:
        a.t_step = cfg.start

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
            order = a.generate_orders()
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
