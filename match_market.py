#!/usr/bin/env python3
from pathlib import Path
from argparse import ArgumentParser
from time import time
import os
import json
import glob
import logging
import warnings

from simply import market, market_2pac, market_fair, market_tarif
from simply.scenario import load, create_random, Scenario
from simply.config import Config
from simply.util import summerize_actor_trading, dates_to_datetime
from simply.simply_main import main


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
