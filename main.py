#!/usr/bin/env python3

from argparse import ArgumentParser

from simply import market, market_2pac, market_fair
from simply.scenario import load, create_random, Scenario
from simply.config import Config
from simply.util import summerize_actor_trading


"""
Entry point for standalone functionality.

Reads in configuration file (or uses defaults when none is supplied),
creates or loads scenario and matches orders in each timestep.
May show network plots or print market statistics, depending on config.

Usage: python main.py [config file]
"""


def main(cfg: Config):

    # Load scenario, if path exists with the correct format
    # otherwise remove all files in existing folder and create new scenario
    # check if actor-files with correct format exist in cfg.path
    scenario_exists = len([False for i in cfg.path.glob(f"actor_*.{cfg.data_format}")]) != 0

    print(f'Scenario path: {cfg.path}')

    # load existing scenario or else create randomized new one
    sc: Scenario
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
        else:
            # create path if it does not exist yet
            cfg.path.mkdir(parents=True, exist_ok=True)
        sc = create_random(cfg.nb_nodes, cfg.nb_actors, cfg.weight_factor)
        sc.save(cfg.path, cfg.data_format)

    if cfg.show_plots:
        sc.power_network.plot()
        sc.plot_participant_data()

    # generate requested market
    if "pac" in cfg.market_type:
        m = market_2pac.TwoSidedPayAsClear(network=sc.power_network)
    elif cfg.market_type in ["fair", "merit"]:
        m = market_fair.BestMarket(network=sc.power_network)
    else:
        # default
        m = market.Market(network=sc.power_network)

    sc.add_market(m)
    for _ in range(cfg.nb_ts):
        # actors calculate strategy based market interaction with the market maker
        sc.create_strategies()

        # orders are generated based on the flexibility towards the planned market interaction
        # and a pricing scheme. Orders are matched at the end
        sc.market_step()

        # actors are prepared for the next time step by changing socs, banks and predictions
        sc.next_time_step()

        if cfg.show_prints:
            print("Matches of bid/ask ids: {}".format(m.matches))
            print(
                "\nCheck individual traded energy blocks (splitted) and price at market level"
            )

    if cfg.show_prints:
        print("\nTraded energy volume and price at actor level")
        print(summerize_actor_trading(sc))

    return sc


if __name__ == "__main__":
    parser = ArgumentParser(description='Entry point for market simulation')
    parser.add_argument('config', nargs='?', default="", help='configuration file')
    args = parser.parse_args()

    cfg = Config(args.config)
    main(cfg)
