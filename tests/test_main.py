from pathlib import Path

from simply.config import Config
from match_market import main


class TestMain:
    def test_main(self, tmp_path):
        cfg = Config("", "")
        cfg.path = Path((tmp_path / "output"))
        current_path = Path.cwd()
        main(cfg)

    def test_load_scenario_csv(self, tmp_path):
        cfg = Config("","")
        # cfg.save_csv = True is the default value. Therefore, we don't set it
        cfg.data_format = "csv"
        cfg.path = Path((tmp_path / "output"))

        main(cfg)

        cfg.load_scenario = True
        main(cfg)

    def test_load_scenario_json(self, tmp_path):
        cfg = Config("","")
        cfg.save_csv = True
        cfg.data_format = "json"
        cfg.path = Path((tmp_path / "output"))
        main(cfg)

        cfg.load_scenario = True
        main(cfg)
