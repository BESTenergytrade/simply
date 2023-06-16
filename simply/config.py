import warnings
from configparser import ConfigParser, MissingSectionHeaderError, NoOptionError, NoSectionError
from numpy import linspace
from pathlib import Path


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
        - step_size - length of timestep in hours [1]\n
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
            fair/merit (custom BEST market)\n
        - energy_unit: size of energy units to be traded individually [0.01]\n
        - weight_factor: conversion factor from grid fees to power network node weight [0.1]\n
    [actor]
        - horizon - number of time steps to look ahead for prediction [24]

    :param cfg_file: configuration file path with the attributes listed above.
    :type cfg_file: str
    :keyword cfg_file: start
    """

    def __init__(self, cfg_file):
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
        try:
            self.project_path = parser.get("default", "project_path")
        except (NoOptionError, NoSectionError):
            raise ValueError("Configuration file must specify the 'project_path' parameter (path of project directory).")
        self.project_path = Path(self.project_path)
        self.results_path = parser.get("default", "results_path", fallback=str(self.project_path / "market_results"))
        self.results_path = Path(self.results_path)
        self.scenario_path = parser.get("default", "scenario_path", fallback=str(self.project_path / "scenario"))
        self.scenario_path = Path(self.scenario_path)
        self.data_format = parser.get("default", "data_format", fallback="json")
        # load existing scenario
        self.load_scenario = parser.getboolean("default", "load_scenario", fallback=True)

        # For generating random scenario
        # number actors in simulation
        #self.nb_actors = parser.getint('default', 'nb_actors', fallback=5)
        self.nb_actors = parser.getint('default', 'nb_actors', fallback=None)
        # number of nodes in simulation
        self.nb_nodes = parser.getint('default', 'nb_nodes', fallback=4)
        # weight factor: network charges to power network weight
        self.weight_factor = parser.getfloat("default", "weight_factor", fallback=0.1)

        # Tolerance value for assertions, comparison and so on
        self.EPS = parser.getfloat("default", "EPS", fallback=1e-6)

        # --------------------------
        # market
        # --------------------------
        # market type to be use
        self.market_type = parser.get("default", "market_type", fallback="default").lower()
        # reset market after each interval (discard unmatched orders)
        self.reset_market = parser.getboolean("default", "reset_market", fallback=True)
        # size of energy units to be traded individually
        self.energy_unit = parser.getfloat("default", "energy_unit", fallback=0.01)
        # default grid_fee to be used by market maker
        self.default_grid_fee = parser.getfloat("default", "default_grid_fee", fallback=0)

        # time related
        # start time step
        self.start = parser.getint("default", "start", fallback=0)
        # number of timesteps in simulation
        self.nb_ts = parser.getint("default", "nb_ts", fallback=3)
        # interval between simulation timesteps
        self.step_size = parser.getint("default", "step_size", fallback=1)
        # list of timesteps in simulation
        # not read from file but created from above information
        self.list_ts = linspace(self.start, self.start + self.nb_ts - 1, self.nb_ts)

        # --------------------------
        # actor
        # --------------------------
        # Horizon up to which energy management is considered and predictions are made
        self.horizon = parser.getint("default", "horizon", fallback=24)
        # strategy that every actor uses
        self.actor_strategy = parser.getint("default", "actor_strategy", fallback=None)

        # --------------------------
        # output
        # --------------------------
        # show various plots
        self.show_plots = parser.getboolean("default", "show_plots", fallback=False)
        # print debug info to console
        self.show_prints = parser.getboolean("default", "show_prints", fallback=False)
        # save orders and matching results to csv files
        self.save_csv = parser.getboolean("default", "save_csv", fallback=False)
