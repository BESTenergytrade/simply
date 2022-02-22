from configparser import ConfigParser, MissingSectionHeaderError
from numpy import linspace
from pathlib import Path


class Config:
    """
    Class holding all simulation-relevant information. Read in from configuration file.

    After creation, the generated instance is available as Config.config,
    the parsed file as Config.parser.

    Config attributes, grouped by section, with default in brackets:
    [default]
    start - initial timestep [8]
    nb_ts - number of timesteps to simulate [3]
    step_size - length of timestep in hours [1]
    list_ts - list of timesteps in simulation [generated, can't be overridden]
    show_plots - show various plots [False]
    show_prints - show debug info in terminal [False]
    save_csv - save orders and mathced results to csv files [True]
    path - path of scenario directory to load and/or store [./scenarios/default]
    data_format - how to save actor data. "csv": save data in separate csv file and all actors
    in one config file, otherwise save config and data per actor in a single file ["cfg"]
    reset_market: if set, discard unmatched orders after each interval [True]
    update_scenario: if set, always save scenario in given path (even if loaded) [False]
    market_type: selects matching strategy. Supported:
        [default]/pab/basic (pay-as-bid)
        pac/2pac (two-sided pay-as-clear)
        fair/merit (custom BEST market)
    weight_factor: conversion factor from grid fees to power network node weight [0.1]
    [actor]
    horizon - number of timesteps to look ahead for prediction [24]
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
        # interval between simulation timesteps
        self.step_size = parser.getint("default", "step_size", fallback=1)
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
