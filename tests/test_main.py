import pytest
from pathlib import Path
import pandas as pd

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


@pytest.fixture
def example_project(project_name):
    return Path(__file__).resolve().parents[1] / "projects/example_projects" / project_name


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # round float columns for numeric stability
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].round(6)

    return df


class TestProjects:
    @pytest.mark.parametrize('project_name', ["example_project", "example_project_ev_opt"])
    def test_example_scenarios(self, example_project):
        proj_dir = example_project
        cfg = Config(proj_dir / "config.cfg", proj_dir)
        # cfg.save_csv = True is the default value. Therefore, we don't set it

        cfg.load_scenario = True  # (already configured that way)
        cfg.show_plots = False
        # test results to other folder
        order_validation = cfg.results_path / f"orders.csv"
        match_validation = cfg.results_path / f"matches.csv"
        output_dir = Path() / f"output_{example_project.name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg.results_path = output_dir
        # tests that example project runs through without errors
        main(cfg)
        # TODO compare results did not change ...
        order_generated = output_dir / f"orders.csv"
        match_generated = output_dir / f"matches.csv"

        # Load and compare orders
        df_order_gen = _load_csv(order_generated)
        df_order_val = _load_csv(order_validation)
        pd.testing.assert_frame_equal(df_order_gen, df_order_val)

        # Load and compare matches
        df_match_gen = _load_csv(match_generated)
        df_match_val = _load_csv(match_validation)
        pd.testing.assert_frame_equal(df_match_gen, df_match_val)
