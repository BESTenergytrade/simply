from configparser import ConfigParser, MissingSectionHeaderError
from numpy import linspace
from pathlib import Path

class Config:

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
        self.start = parser.getint("default", "start", fallback = 8)
        # number of timesteps in simulation
        self.nb_ts = parser.getint("default", "nb_ts", fallback = 3)
        # interval between simulation timesteps
        self.step_size = parser.getint("default", "step_size", fallback = 1)
        # list of timesteps in simualtion
        # not read from file but created from above information
        self.list_ts = linspace(self.start, self.start + self.nb_ts - 1, self.nb_ts)

        # show various plots
        self.show_plots = parser.getboolean("default", "show_plots", fallback=False)
        # print debug info to console
        self.show_prints = parser.getboolean("default", "show_prints", fallback=False)

        # path of scenario file to load and/or store
        self.path = parser.get("default", "path", fallback = "./scenarios/default")
        self.path = Path(self.path)
        # reset market after each interval (discard unmatched orders)
        self.reset_market = parser.getboolean("default", "reset_market", fallback=True)
        # always create new scenario in given path
        self.update_scenario = parser.getboolean("default", "update_scenario", fallback=False)

        # which market type to use?
        # supported:
        #   pab/basic/default (pay-as-bid)
        #   pac/2pac (two-sided pay-as-clear)
        #   fair/merit (custom BEST market)
        self.market_type = parser.get("default", "market_type", fallback="default").lower()
