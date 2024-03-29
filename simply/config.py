import warnings
from configparser import ConfigParser, MissingSectionHeaderError
from numpy import linspace
from pathlib import Path
from math import log
import json


class Config:
    """
    Class holding all simulation-relevant information. Read in from configuration file.

    After creation, the generated instance is available as Config.config,
    the parsed file as Config.parser.

    Config attributes, grouped by section, with default in brackets

    [default]
        - start - initial timestep [8]\n
        - nb_ts - number of timesteps to simulate [3]\n
        - nb_actors - number of actors in network [5]\n
        - nb_nodes - number of nodes in network [4]\n
        - ts_per_hour - number of timesteps within one hour [4]\n
        - list_ts - list of timesteps in simulation [generated, can't be overridden]\n
        - show_plots - show various plots [False]\n
        - show_prints - show debug info in terminal [False]\n
        - save_csv - save orders and matched results to csv files [True]\n
        - path - path of scenario directory to load and/or store [./scenarios/default]\n
        - data_format - how to save actor data. Supported values\n
            csv: save data in separate csv file and all actors in one config file,\n
            [json]: save config and data per actor in a single file\n
        - reset_market: if set, discard unmatched orders after each interval [True]\n
        - update_scenario: if set, always save scenario in given path (even if loaded) [False]\n
        - market_type: selects matching strategy. Supported values\n
            [pab]/basic (pay-as-bid)\n
            pac/2pac (two-sided pay-as-clear)\n
            fair (custom BEST market)\n
        - energy_unit: size of energy units to be traded individually [0.01]\n
        - weight_factor: conversion factor from grid fees to power network node weight [0.03]\n
    [actor]
        - horizon - number of time steps to look ahead for prediction [24]

    :param cfg_file: configuration file path with the attributes listed above.
    :type cfg_file: str
    :keyword cfg_file: start
    """

    def __init__(self, cfg_file, project_dir):
        global config
        config = self
        global parser
        parser = ConfigParser()
        # ToDo: probably change this to an error, because default values will not be used
        if not cfg_file:
            warnings.warn("No Configuration file path was provided. Default values will be used.")
        elif not Path(cfg_file).is_file():
            warnings.warn(f"{cfg_file} was provided as Configuration file, but this file does not "
                          "exist. Default values will be used.")

        if not project_dir:
            warnings.warn("No project_dir was provided. Default project_dir ./projects/"
                          "example_projects/example_project is used")
            project_dir = "projects/example_projects/example_project"
        elif not Path(project_dir):
            warnings.warn(f"{project_dir} was provided as directory, but this directory does not "
                          f"exist. Default project_dir ./projects/example_project will be used.")
            project_dir = "projects/example_projects/example_project"

        try:
            parser.read(cfg_file)
        except MissingSectionHeaderError:
            # headless config: insert missing section header
            with open(cfg_file, 'r') as f:
                config_string = "[default]\n" + f.read()
            parser.read_string(config_string)

        # default section: basic simulation properties
        # --------------------------
        # scenario
        # --------------------------
        self.project_path = Path(project_dir)
        self.scenario_path = parser.get("default", "scenario_path", fallback=str(self.project_path /
                                                                                 "scenario"))
        self.scenario_path = Path(self.scenario_path)
        self.data_format = parser.get("default", "data_format", fallback="json")
        self.buy_sell_lin_param = json.loads(
            parser.get("default", "buy_sell_lin_param", fallback="[0, 1]"))
        # load existing scenario
        self.load_scenario = parser.getboolean("default", "load_scenario", fallback=False)

        # For generating random scenario
        # number actors in simulation
        self.nb_actors = parser.getint('default', 'nb_actors', fallback=5)
        # number of nodes in simulation
        self.nb_nodes = parser.getint('default', 'nb_nodes', fallback=4)

        # Tolerance value for assertions, comparison and so on
        self.EPS = parser.getfloat("default", "EPS", fallback=1e-6)
        self.round_decimal = round(log(1 / self.EPS, 10))

        # --------------------------
        # market
        # --------------------------
        # market type to be use
        self.market_type = parser.get("default", "market_type", fallback="default").lower()
        self.disputed_matching = parser.get("default", "disputed_matching",
                                            fallback="grid_fee").lower()
        # reset market after each interval (discard unmatched orders)
        self.reset_market = parser.getboolean("default", "reset_market", fallback=True)
        # size of energy units to be traded individually
        self.energy_unit = parser.getfloat("default", "energy_unit", fallback=0.01)
        # factor describing the relation of grid fee to cumulative power network edge weights
        self.weight_factor = parser.getfloat("default", "weight_factor", fallback=0.03)
        # default grid_fee to be used by market maker
        self.default_grid_fee = parser.getfloat("default", "default_grid_fee", fallback=0)
        # local grid fee to be used
        self.local_grid_fee = parser.getfloat("default", "local_grid_fee", fallback=0)

        # time related
        # start date of time series data
        self.start_date = parser.get("default", "start_date", fallback="2016-01-01")
        # start time step
        self.start = parser.getint("default", "start", fallback=0)
        # number of timesteps in simulation
        self.nb_ts = parser.getint("default", "nb_ts", fallback=5)
        # number of timesteps within one hour
        self.ts_per_hour = parser.getint("default", "ts_per_hour", fallback=4)
        # list of timesteps in simulation
        # not read from file but created from above information
        self.list_ts = linspace(self.start, self.start + self.nb_ts - 1, self.nb_ts)

        # --------------------------
        # actor
        # --------------------------
        # Horizon up to which energy management is considered and predictions are made
        self.horizon = parser.getint("default", "horizon", fallback=24)
        # strategy that every actor uses
        self.actor_strategy = parser.getint("default", "actor_strategy", fallback=0)

        # --------------------------
        # output
        # --------------------------
        # show various plots
        self.show_plots = parser.getboolean("default", "show_plots", fallback=False)
        # print warning info to console
        self.verbose = parser.getboolean("default", "verbose", fallback=False)
        # print debug info to console
        self.debug = parser.getboolean("default", "debug", fallback=False)
        # print intermediate results info to console
        self.show_prints = parser.getboolean("default", "show_prints", fallback=False)
        # save orders and matching results to csv files
        self.save_csv = parser.getboolean("default", "save_csv", fallback=False)
        self.results_path = parser.get("default", "results_path", fallback=str(self.project_path /
                                                                               "market_results"))
        self.results_path = Path(self.results_path)
