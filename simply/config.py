from configparser import ConfigParser, MissingSectionHeaderError
from numpy import linspace
from pathlib import Path

class Config:

    def __init__(self, cfg_file):
        global parser
        parser = ConfigParser()
        try:
            parser.read(cfg_file)
        except MissingSectionHeaderError:
            # headless config: insert missing section header
            with open(cfg_file, "r") as f:
                config_string = '[default]\n' + f.read()
            parser.read_string(config_string)
        self.start = parser.getint("default", "start", fallback = 8)
        self.nb_ts = parser.getint("default", "nb_ts", fallback = 3)
        self.step_size = parser.getint("default", "step_size", fallback = 1)
        self.show_plots = parser.getboolean("default", "show_plots", fallback=False)
        self.show_prints = parser.getboolean("default", "show_prints", fallback=False)
        self.update_scenario = parser.getboolean("default", "update_scenario", fallback=False)
        self.path = parser.get("default", "path", fallback = './scenarios/default')
        self.path = Path(self.path)

        self.list_ts = linspace(self.start, self.start + self.nb_ts - 1, self.nb_ts)
