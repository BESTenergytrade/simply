import datetime
import os
import json
import shutil
import warnings

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from pandas.api.types import is_numeric_dtype
from simply.actor import Actor
from simply.scenario import Scenario
from simply.power_network import create_power_network_from_config
from simply.config import Config

"""
This script creates a simulation scenario from a config JSON file, network JSON file,
configuration text file, a loads assignment CSV file, and a data directory.

It creates a scenario object and uses this to create an Actor and Power Network object. The Actor
object is created using a basic strategy function, and the Power Network is created using a
create power network from config function. Additionally, it has helper functions to insert the
market maker ID, check if the data is present, remove existing scenario directory,
convert string dates to datetime dtype, and build a pandas dataframe from the config JSON.
"""


# Helper functions
def get_mm_prices(dirpath, start_date, end_date):
    csv_df = pd.read_csv(dirpath, sep=',', parse_dates=['Time'], dayfirst=False,
                         index_col=['Time'])
    # Make sure dates are parsed
    csv_df.index = pd.to_datetime(csv_df.index)
    try:
        return list(csv_df.loc[start_date:end_date]["prices"])
    except KeyError:
        # is first column after time column numeric?
        if is_numeric_dtype(csv_df.loc[start_date:end_date].iloc[:, 0]):
            # if so, we assume that is the price column even though its not named "prices"
            warnings.warn("Prices data file does not contain column named 'prices'. Instead the "
                          f"first column named {csv_df.iloc[:, 0].name} is used")
            return list(csv_df.loc[start_date:end_date].iloc[:, 0])
        else:
            raise Exception("Prices data file does not contain column named 'prices' and the "
                            "second column is not numeric, which would be used otherwise.")


def insert_market_maker_id(dirpath):
    """Changes market_maker buy and sell to shared id within the actors json path of the dirpath."""
    with open(f'{dirpath}/actors.json') as f:
        d = json.load(f)
    if 'market_maker_buy' in d:
        d['market_maker_buy']['id'] = 'market_maker'
        d['market_maker_sell']['id'] = 'market_maker'
        dirpath.mkdir(parents=True, exist_ok=True)
        dirpath.joinpath('actors.json').write_text(json.dumps(d, indent=2))


def check_data_present(loads_path, pv_path, price_path):
    """Returns a custom error message if load, pv or price data files are missing."""
    for path in [loads_path, pv_path, price_path]:
        if len(os.listdir(path)) == 0:
            raise Exception(f'{path} is missing data.')


def remove_existing_dir(path):
    """Deletes existing repository stored in scenario save location. """
    if path.is_dir():
        shutil.rmtree(path)


def dates_to_datetime(start_date="2016-01-01", nb_ts=None, horizon=24, ts_hour=1):
    """Converts string dates to datetime dtype and calculates end date from timesteps parameter."""
    start_date = pd.to_datetime(start_date)
    time_change = datetime.timedelta(minutes=(nb_ts + horizon - 1) * (60 / ts_hour))
    end_date = start_date + time_change
    return start_date, end_date


def basic_strategy(df, csv_peak, ps, ls):
    """Scales load and pv by peak or scaling factor parameter and calculates schedule."""
    if ls:
        df['load'] *= ls
    else:
        df['load'] *= csv_peak['load']
    if 'pv' in csv_peak:
        if ps:
            df['pv'] *= ps
        else:
            df['pv'] *= csv_peak['pv']
    # remove nan
    df = df.fillna(0)
    df["schedule"] = df["pv"] - df["load"]
    return df


def map_actors(config_df):
    """Helper function to build power network from community config json."""
    map = {}
    for i in config_df.index:
        map[config_df["prosumerName"][i]] = str(config_df["gridLocation"][i])
    return map


def read_config_json(config_json):
    """Builds a pandas dataframe containing values from config json and splits market_maker into
    buy and sell."""
    try:
        config_df = pd.read_json(config_json)
    except ValueError as e:
        raise ValueError(f"You have to provide a correct json file: {e}")
    if 'devices' not in config_df:
        config_df['devices'] = np.nan
    # Do not include market maker

    return config_df


def create_actor_from_config(actor_id, environment, asset_dict={}, start_date="2016-01-01",
                             nb_ts=None, horizon=24, ts_hour=1,
                             cols=["load", "pv", "schedule", "price"], ps=None, ls=None,
                             strategy=0, pricing_strategy=None):
    """
    Create Actor with an ID and given asset time series shifted to a specified start time and
    resolution (and scaled by factors ps/ls if given).

    :param actor_id: ID of the actor
    :param asset_dict: Dictionary of asset information
    :param start_date: Start date of the actor, defaults to "2016-01-01"
    :param nb_ts: Number of time slots to be generated, defaults to None
    :param horizon: number of time slots to look into future to make the prediction for actor
        strategy
    :param ts_hour: Number of time slot of equal length within one hour, defaults to 4
    :param cols: List of columns to be included, defaults to ["load", "pv", "schedule", "price"]
    :param ps: PV scalar, defaults to None
    :param ls: Load scalar, defaults to None

    :return: Actor object
    """
    df = pd.DataFrame([], columns=cols)
    start_date, end_date = dates_to_datetime(start_date, nb_ts + 1, horizon, ts_hour)
    # Read csv files for each asset
    csv_peak = {}
    battery_cap = 0
    init_soc = 0.5
    for col, info_dict in asset_dict.items():
        # if info_dict is empty
        if not info_dict:
            continue
        if col == "battery":
            battery_cap = info_dict["capacityKwh"]
            init_soc = info_dict["initialSOC"]
            continue
        csv_df = pd.read_csv(info_dict["csv"], sep=',', parse_dates=['Time'], dayfirst=False,
                             index_col=['Time'])
        # Make sure dates are parsed
        csv_df.index = pd.to_datetime(csv_df.index)
        if csv_df.index[-1] < end_date:
            raise IndexError(f"Provided input data ({csv_df.index[-1]} + {int(60 / ts_hour)} min) "
                             f"ends before configured ending time {end_date} resulting of config"
                             f"parameters:"
                             f"start_date + (nb_ts + horizon + 1) * (60 / ts_hour) min.")

        df.loc[:, col] = csv_df.loc[start_date:end_date].iloc[:, 0]
        # Save peak value and normalize time series
        csv_peak[col] = df[col].max()
        df[col] = df[col] / csv_peak[col]

    df = basic_strategy(df, csv_peak, ps, ls)

    return Actor(actor_id, df, environment, ls=1, ps=1, battery_cap=battery_cap,
                 battery_initial_soc=init_soc, strategy=strategy, pricing_strategy=pricing_strategy)


def create_scenario_from_config(config_json, network_path, loads_dir_path, data_dirpath=None,
                                buy_sell_function=None,
                                weight_factor=1, ts_hour=4, nb_ts=None, horizon=24,
                                start_date=None, plot_network=False,
                                price_filename="basic_prices.csv", ps=None, ls=None):
    """
    Create Scenario object while creating Actor objects from config_json referencing to time series
     data in data_path. The Actors are further mapped to a defined network.

    :param config_json: Path object of the configuration json file
    :param network_path: Path object of the network json file
    :param loads_dir_path: Path object of the directory containing loads csv
    :param data_dirpath: Path object of the directory containing all time series data,
        defaults to None
    :param weight_factor: Weight factor used to derive grid fees, defaults to 1
    :param ts_hour: Number of time slot of equal length within one hour, defaults to 4
    :param nb_ts: Number of time slots to be generated, defaults to None
    :param horizon: number of time slots to look into future to make the prediction for actor
        strategy
    :param start_date: Start date of the scenario,defaults to "2016-01-01"
    :param plot_network: Boolean value to indicate whether the network should be plotted,
        defaults to False
    :param price_filename: Name of the price csv file, defaults to "basic_prices.csv"
    :param ps: PV scalar, defaults to None
    :param ls: Load scalar, defaults to None
    :return: Scenario object
    """
    # Extend paths
    loads_path = data_dirpath.joinpath("load")
    pv_path = data_dirpath.joinpath("pv")
    price_path = data_dirpath.joinpath("price")

    # check for data
    check_data_present(loads_path, pv_path, price_path)

    # Parse json
    config_df = read_config_json(config_json)

    # Create nodes for power network
    pn = create_power_network_from_config(network_path, weight_factor)

    if plot_network is True:
        pn.plot()

    if start_date is None:
        start_date = "2016-01-01"
        warnings.warn(f"No start date was given, use default date {start_date}.")
    start_date, end_date = dates_to_datetime(start_date, nb_ts + 1, horizon, ts_hour)
    buy_prices = get_mm_prices(price_path / price_filename, start_date, end_date)
    # Empty scenario. Member Participants, map actors and power network will be added later
    # When buy_prices are provided a market maker is automatically generated
    scenario = Scenario(None, None, buy_prices=buy_prices, buy_to_sell_function=buy_sell_function)
    for i, actor_row in config_df.iterrows():
        file_dict = {}
        asset_dict = {}
        # If there is no devices use
        if actor_row['devices'] != actor_row['devices']:
            file_df = pd.read_csv(loads_dir_path)
            current_type_files = file_df[file_df['Type'] == actor_row['prosumerType']]
            # Take the first filename associated with the prosumerType
            file_dict['load'] = list(current_type_files['Filename'])[0]
        else:
            for device in actor_row['devices']:
                # save the csv file name
                if device['deviceType'] == 'battery':
                    if 'battery' in asset_dict.keys():
                        warnings.warn(
                            f"Actor{actor_row['prosumerName']} has multiple battery devices.")
                    device.pop('deviceType')
                    asset_dict['battery'] = device
                else:
                    file_dict[device['deviceType']] = device['deviceID']

        # Load
        if 'load' in file_dict:
            if 'load' in asset_dict.keys():
                warnings.warn(f"Actor{actor_row['prosumerName']} has multiple load devices.")
            asset_dict['load'] = {"csv": loads_path.joinpath(file_dict['load']), "col_index": 1}

        # PV
        if 'solar' in file_dict:
            if 'pv' in asset_dict.keys():
                warnings.warn(f"Actor{actor_row['prosumerName']} has multiple solar devices.")
            asset_dict['pv'] = {"csv": pv_path.joinpath(file_dict['solar']), "col_index": 1}

        # Prices
        asset_dict['price'] = {"csv": price_path.joinpath(price_filename), "col_index": 1}
        # actors are automatically added to the scenario environment
        _ = create_actor_from_config(actor_row['prosumerName'], scenario.environment,
                                     asset_dict=asset_dict, start_date=start_date,
                                     nb_ts=nb_ts, horizon=horizon, ts_hour=ts_hour, ps=ps, ls=ls,
                                     strategy=actor_row.get('strategy'),
                                     pricing_strategy=actor_row.get("pricing_strategy"))
        print(f'- Added Actor ({i}) {actor_row["prosumerName"]}: "{file_dict["load"]}"')

    actor_map = map_actors(config_df)
    actor_map = pn.add_actors_map(actor_map)

    if plot_network is True:
        pn.plot()
        pn.to_json()

    # Update shortest paths and the grid fee matrix
    pn.update_shortest_paths()
    pn.generate_grid_fee_matrix(weight_factor)
    scenario.power_network = pn
    scenario.actor_map = actor_map
    return scenario


def main(project_dir, data_dir):
    project_dir = Path(project_dir)
    # Set the paths based on the scenario directory
    config_json_path = project_dir / "actors_config.json"
    network_path = project_dir / "network_config.json"
    config_path = project_dir / "config.cfg"
    data_dirpath = Path(data_dir) if data_dir else project_dir / "scenario_inputs"
    loads_dir_path = data_dirpath / "loads_dir.csv"

    if not config_path.exists():
        # backwards compatibility to allow config.txt
        config_path = config_path.parent / (config_path.stem + '.txt')

    missing_paths = [path for path in (config_json_path, network_path, config_path, loads_dir_path,
                                       data_dirpath) if not path.exists()]
    if missing_paths:
        missing_paths_str = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            f"One or more required files do not exist in the scenario directory:"
            f"\n{missing_paths_str}")

    # Build config object using configuration file
    cfg = Config(config_path, project_dir)

    # Set cfg.path based on scenario_path from config file or default to project_dir/"scenario"
    scenario_path = cfg.scenario_path if hasattr(cfg, 'scenario_path') else None
    cfg.path = Path(scenario_path) if scenario_path else project_dir / "scenario"

    # Reset save location directory
    remove_existing_dir(cfg.path)

    sc = create_scenario_from_config(
        config_json_path,
        network_path,
        weight_factor=cfg.weight_factor,
        data_dirpath=data_dirpath,
        buy_sell_function=lin_parameter_function(cfg.buy_sell_lin_param),
        start_date=cfg.start_date,
        nb_ts=cfg.nb_ts,
        horizon=cfg.horizon,
        ts_hour=cfg.ts_per_hour,
        loads_dir_path=loads_dir_path,
        ps=None,
        ls=None
    )
    sc.save(cfg.path, cfg.data_format)
    # insert_market_maker_id(cfg.path)

    if cfg.show_plots:
        sc.power_network.plot()
        sc.plot_participant_data()


def lin_parameter_function(p):
    assert len(p) == 2
    return lambda x: p[0] + x * p[1]


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Entry point for market simulation')
    parser.add_argument('project_dir', help='project directory path')
    parser.add_argument('--data_dir', default='', help='data directory')
    args = parser.parse_args()

    if args.project_dir is None:
        raise FileNotFoundError(
            "Project directory path must be specified. Please provide the path as a command-line "
            "argument.")
    data_dir = args.data_dir if args.data_dir is not None else os.path.join(args.project_dir,
                                                                            "scenario_inputs")
    if args.data_dir is None:
        print(f"Using data directory: {data_dir}")

    # Call the main function with the specified scenario directory and data directory
    main(args.project_dir, args.data_dir)
