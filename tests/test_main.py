import pytest

from simply.config import Config
from match_market import main
from simply.market_maker import MarketMaker


class TestMain:
    def test_main(self, tmp_path):
        with pytest.raises(AttributeError):
            cfg = Config("")
            # missing path
            main(cfg)

        cfg = Config("", tmp_path)
        main(cfg)

    def test_load_scenario_csv(self, tmp_path):
        cfg = Config("", tmp_path)
        # cfg.save_csv = True is the default value. Therefore, we don't set it
        cfg.data_format = "csv"
        sc = main(cfg)

        cfg.load_scenario = True
        sc_loaded = main(cfg)

        # check that loaded energy values are equal to the ones generated before
        for i, p in enumerate(sc.market_participants):
            if not isinstance(p, MarketMaker):
                p.data.equals(sc_loaded.market_participants[i].data)
            else:
                pass

    def test_load_scenario_json(self, tmp_path):
        cfg = Config("", tmp_path)
        cfg.data_format = "json"
        main(cfg)

        cfg.load_scenario = True
        main(cfg)

    def test_save_results(self, tmp_path):
        cfg = Config("", tmp_path)
        cfg.save_csv = True
        main(cfg)
