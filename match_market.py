#!/usr/bin/env python3
from pathlib import Path
from argparse import ArgumentParser
from time import time
import os
import json
import glob
import logging

from simply import market, market_2pac, market_fair, market_tarif
from simply.defaults import MARKETID
from simply.scenario import load, create_random, Scenario
from simply.config import Config
from simply.util import summerize_actor_trading, dates_to_datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
"""
Entry point for standalone functionality.

Reads in configuration file (or uses defaults when none is supplied),
creates or loads scenario and matches orders in each timestep.
May show network plots or print market statistics, depending on config.

Usage: python match_market.py [config file]
"""


def main(cfg: Config):
    # Checks if actor files with the correct format exist in the cfg.scenario_path
    # --------------------------------
    def list_files_in_path(current_path, pattern='*'):
        file_list = glob.glob(os.path.join(current_path, pattern))
        return file_list

    current_path = Path.cwd()

    print("Current directory:", current_path)
    files_in_path = list_files_in_path(cfg.scenario_path)
    print(f"Files in {cfg.scenario_path}:  {len(files_in_path)}")
    print("data_format: ", cfg.data_format)
    # --------------------------------

    scenario_exists = len(
        [False for i in cfg.scenario_path.glob(f"*actor*_*.{cfg.data_format}")]) != 0
    print("scenario_exists: ", scenario_exists)

    markets_json = cfg.scenario_path / "markets.json"
    print("markets_json exists: ", markets_json.is_file())

    # load existing scenario or else create randomized new one
    sc: Scenario

    if cfg.load_scenario:
        print(f"Load scenario from cfg.scenario_path: {cfg.scenario_path}")
        if scenario_exists:
            sc = load(cfg.scenario_path, cfg.data_format)
        else:
            raise Exception(
                f'Could not find scenario path: {cfg.scenario_path}. Make sure to include the '
                f'scenario directory in your project or if you want to generate a random scenario, '
                f'set load_scenario = False in config.cfg.')
    else:
        print(f"Create scenario at cfg.scenario_path: {cfg.scenario_path}")
        if cfg.scenario_path.exists():

            raise Exception(
                f'The path: {cfg.scenario_path} already exists with another file structure. '
                'Please remove or rename folder to avoid confusion and restart '
                'simulation.')
        else:
            # create scenario path if it does not exist yet
            cfg.scenario_path.mkdir(parents=True, exist_ok=True)
        sc = create_random(cfg.nb_nodes, cfg.nb_actors, cfg.weight_factor)
        sc.save(cfg.scenario_path, cfg.data_format)

    start_date, end_date, time_range = dates_to_datetime(cfg.start_date, cfg.nb_ts + 1, cfg.horizon,
                                                         cfg.ts_per_hour)

    if cfg.show_plots:
        sc.power_network.plot()
        sc.plot_participant_data()
        sc.plot_prices()

    # generate requested market(s)
    if markets_json.is_file():
        with open(markets_json) as f:
            market_configs = json.load(f)
    else:
        market_configs = [{"market_type": cfg.market_type,
                          "market_name": MARKETID,
                          "disputed_matching": cfg.disputed_matching}]
    for mc in market_configs:
        if "pac" in mc["market_type"]:
            m = market_2pac.TwoSidedPayAsClear(name=mc["market_name"], network=sc.power_network)
        elif "fair" in mc["market_type"]:
            m = market_fair.BestMarket(name=mc["market_name"],
                                       network=sc.power_network,
                                       disputed_matching=mc["disputed_matching"])
        elif "pab" in mc["market_type"]:
            # default pay-as-bid
            m = market.Market(name=mc["market_name"])
        elif "tarif" in mc["market_type"]:
            m = market_tarif.MarketMakerDirectTarif(name=mc["market_name"], network=sc.power_network)
        else:
            raise NotImplementedError(
                "This matching algorithm is not implemented, choose out of: ['pab', 'pac', 'fair']")
        sc.add_to_market_dict(mc["market_name"], m)

    exec_start = time()

    for i, t in enumerate(time_range[cfg.start:cfg.nb_ts]):
        # actors calculate strategy based market interaction with the market maker
        sc.create_strategies(update_step=cfg.schedule_update_step)
        logging.info("Actors finished scheduling created")

        # orders are generated based on the flexibility towards the planned market interaction
        # and a pricing scheme. Orders are matched at the end
        sc.market_step()

        # actors are prepared for the next time step by changing socs, banks and predictions
        sc.next_time_step()
        for m in sc.market_dict.values():
            logging.info(f"Cleared Volume: {round(m.cleared_volume[t], cfg.round_decimal)}")

        # save/update additional actor results every at least 10 time steps
        if cfg.save_csv and i % 10 == 0:
            for m in sc.market_dict.values():
                sc.save_additional_results(m.csv_path)
            # currently only debug function (no configuration needed)
            # sc.track_actor_schedule(sc.market.csv_path, actor_id="building_2275985")

    print(f"Total execution time was: {time()-exec_start} s")

    if cfg.show_prints:
        print("Matches of bid/ask ids: {}".format(m.matches))
        print(
            "\nCheck individual traded energy blocks (splitted) and price at market level"
        )
        print("\nTraded energy volume and price at actor level")
        print(summerize_actor_trading(sc))

    # save additional results
    if cfg.save_csv:
        for m in sc.market_dict.values():
            sc.save_additional_results(m.csv_path)
    for m in sc.market_dict.values():
        print(f"Results saved to {m.csv_path}")

    return sc


if __name__ == "__main__":
    parser = ArgumentParser(description='Entry point for market simulation')
    # parser.add_argument('config', nargs='?', default="", help='configuration file')
    # Replaced the above line to take in the project directory (which will contain the config file)
    # instead of putting in the config file
    # also made it mandatory
    parser.add_argument('project_dir', nargs='?', default=None, help='project directory path')
    args = parser.parse_args()
    # Raise error if project directory not specified
    if args.project_dir is None:
        raise FileNotFoundError(
            "Project directory path must be specified. Please provide the path as a command-line "
            "argument, e.g. './projects/example_projects/example_project'. This example "
            "also provides the expected structure of a project.")
    if not Path(args.project_dir).exists():
        raise FileNotFoundError(
                f"The provided project_dir '{args.project_dir}' does not exist.")
    # This means that the config file must always be in the project directory
    config_file = os.path.join(args.project_dir, "config.cfg")
    # Raise error if config.(cfg|txt) file not found in project directory
    if not os.path.isfile(config_file):
        config_file = os.path.join(args.project_dir, "config.txt")
        if not os.path.isfile(config_file):
            raise FileNotFoundError(
                "Config file 'config.cfg' or 'config.txt' not found in project directory: "
                f"{args.project_dir}")

    cfg = Config(config_file, args.project_dir)
    main(cfg)
