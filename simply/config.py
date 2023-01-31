from configparser import ConfigParser, MissingSectionHeaderError
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
        - weight_factor: conversion factor from grid fees to power network node weight [0.1]\n
    [actor]
        - horizon - number of timesteps to look ahead for prediction [24]

    :param cfg_file: configuration file path with the attributes listed above.
    :type cfg_file: str
    :keyword cfg_file: start
    """

    def __init__(self, cfg_file):
        global config
        config = self
        global parser
        parser = ConfigParser()
        try:
            parser.read(cfg_file)
        except MissingSectionHeaderError:
            # headless config: insert missing section header
            with open(cfg_file, 'r') as f:
                config_string = "[default]\n" + f.read()
            parser.read_string(config_string)

        # default section: basic simulation properties
        # start timestep
        self.start = parser.getint("default", "start", fallback=8)
        # number of timesteps in simulation
        self.nb_ts = parser.getint("default", "nb_ts", fallback=3)
        # number actors in simulation
        self.nb_actors = parser.getint('default', 'nb_actors', fallback=5)
        # number of nodes in simulation
        self.nb_nodes = parser.getint('default', 'nb_nodes', fallback=4)
        # number of timesteps per hour
        self.time_steps_per_hour = parser.getint("default", "time_steps_per_hour", fallback=1)
        # list of timesteps in simulation
        # not read from file but created from above information
        self.list_ts = linspace(self.start, self.start + self.nb_ts - 1, self.nb_ts)

        # show various plots
        self.show_plots = parser.getboolean("default", "show_plots", fallback=False)
        # print debug info to console
        self.show_prints = parser.getboolean("default", "show_prints", fallback=False)
        # save orders and matching results to csv files
        self.save_csv = parser.getboolean("default", "save_csv", fallback=True)

        # path of scenario file to load and/or store
        self.path = parser.get("default", "path", fallback="./scenarios/default")
        self.path = Path(self.path)
        self.data_format = parser.get("default", "data_format", fallback="json")
        # reset market after each interval (discard unmatched orders)
        self.reset_market = parser.getboolean("default", "reset_market", fallback=True)
        # always create new scenario in given path
        self.update_scenario = parser.getboolean("default", "update_scenario", fallback=False)

        # which market type to use?
        self.market_type = parser.get("default", "market_type", fallback="default").lower()
        # weight factor: network charges to power network weight
        self.weight_factor = parser.getfloat("default", "weight_factor", fallback=0.1)
        # default grid_fee to be used by market maker
        self.default_grid_fee = parser.getfloat("default", "default_grid_fee", fallback=0)

        # actor section
        self.horizon = parser.getint("default", "horizon", fallback=24)
