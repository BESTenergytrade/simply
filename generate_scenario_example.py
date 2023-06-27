from argparse import ArgumentParser
from pathlib import Path

from simply import scenario
from simply.config import Config

if __name__ == "__main__":
    parser = ArgumentParser(description='Entry point for market simulation')
    parser.add_argument('config', nargs='?', default="", help='configuration file')
    args = parser.parse_args()

    cfg = Config(args.config)
    sc = scenario.create_scenario_from_csv(
        Path("sample", "households_sample"),
        5,
        12,
        0.01,
        nb_ts=3*96
    )
    sc.save(cfg.path, cfg.data_format)

    if cfg.show_plots:
        sc.power_network.plot()
        sc.plot_participant_data()
    sc.power_network.to_image()
    sc.power_network.to_json()
    if cfg.show_prints:
        print(sc.to_dict())
        print(sc.power_network.short_paths)
        print(sc)
