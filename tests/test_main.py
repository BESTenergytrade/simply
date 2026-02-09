import pytest
from pathlib import Path

from simply.config import Config
from match_market import main
from simply.market_maker import MarketMaker
import build_scenario


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
                p_loaded = [a for a in sc_loaded.market_participants if a.id == p.id]
                assert len(p_loaded) == 1
                p.data.equals(p_loaded[0].data)
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


@pytest.fixture
def example_project(project_name):
    return Path(__file__).resolve().parents[1] / "projects/example_projects" / project_name


class TestProjects:
    @pytest.mark.parametrize('project_name', [
        # "example_project",  # scenario to be updated
        # "example_project_ev_opt",  # scenario to be updated
    ])
    def test_running_example_scenarios(self, example_project):
        proj_dir = example_project
        cfg = Config(proj_dir / "config.cfg", proj_dir)
        # cfg.save_csv = True is the default value. Therefore, we don't set it

        cfg.load_scenario = True
        cfg.show_plots = False
        # tests that example project runs through without errors
        main(cfg)

    @pytest.mark.parametrize('project_name', [
        "example_project",
        "example_project_ev_opt",
        "example_project_mmm"
    ])
    def test_build_n_run_example_scenarios(self, example_project):
        proj_dir = example_project
        cfg_path = proj_dir / "config.cfg"
        data_dir = proj_dir / "scenario_inputs"

        cfg = Config(cfg_path, proj_dir)
        cfg.show_plots = False
        # cfg.save_csv = True is the default value. Therefore, we don't have to set it

        # build the scenario
        build_scenario.main(proj_dir, data_dir, cfg_path)

        cfg.load_scenario = True

        # tests that example project runs through without errors
        main(cfg)
