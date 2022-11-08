import datetime
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

from simply.actor import Actor
from simply.scenario import Scenario
from simply.power_network import create_power_network_from_config
from simply.config import Config


def update_actor_json(dirpath):
    with open(f'{dirpath}/actors.json') as f:
        d = json.load(f)
    if 'market_maker_buy' in d:
        d['market_maker_buy']['id'] = 'market_maker'
        d['market_maker_sell']['id'] = 'market_maker'
        dirpath.mkdir(parents=True, exist_ok=True)
        dirpath.joinpath('actors.json').write_text(json.dumps(d, indent=2))


def check_data_present(loads_path, pv_path, price_path):
    for path in [loads_path, pv_path, price_path]:
        if len(os.listdir(path)) == 0:
            raise Exception(f'{path} is missing data.')


def dates_to_datetime(start_date="2016-01-01", nb_ts=None, ts_hour=1):
    start_date = pd.to_datetime(start_date)
    time_change = datetime.timedelta(minutes=(nb_ts - 1) * (60 / ts_hour))
    end_date = start_date + time_change
    return start_date, end_date


def basic_strategy(df, csv_peak, ps, ls):
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


# Helper function to build power network from community config json
def map_actors(config_df):
    map = {}
    for i in config_df.index:
        map[config_df["prosumerName"][i]] = config_df["gridLocation"][i]
    return map


# Actor
def create_actor_from_config(actor_id, asset_dict={}, start_date="2016-01-01", nb_ts=None,
                             ts_hour=1, cols=["load", "pv", "schedule", "prices"],
                             ps=None, ls=None):
    df = pd.DataFrame([], columns=cols)
    start_date, end_date = dates_to_datetime(start_date, nb_ts, ts_hour)

    # Read csv files for each asset
    csv_peak = {}
    for col, csv_dict in asset_dict.items():
        # if csv_dict is empty
        if not csv_dict:
            continue
        csv_df = pd.read_csv(csv_dict["csv"], sep=',', parse_dates=['Time'], dayfirst=True,
                             index_col=['Time'])
        df[col] = csv_df[start_date:end_date]
        # Save peak value and normalize time series
        csv_peak[col] = df[col].max()
        df[col] = df[col] / csv_peak[col]

    df = basic_strategy(df, csv_peak, ps, ls)

    return Actor(actor_id, df, ls=1, ps=1)


def read_config_json(config_json):
    config_df = pd.read_json(config_json)
    if 'devices' not in config_df:
        config_df['devices'] = np.nan
    # Include market maker
    if 'market_maker' in list(config_df['prosumerType']):
        config_df = config_df[config_df.prosumerType != 'market_maker']
        market_maker_sell = pd.DataFrame({'prosumerName': 'market_maker_sell',
                                          'prosumerType': 'market_maker_sell', 'gridLocation':
                                              'market_maker'}, index=[0])
        market_maker_buy = pd.DataFrame({'prosumerName': 'market_maker_buy',
                                         'prosumerType': 'market_maker_buy',
                                         'gridLocation': 'market_maker'}, index=[0])
        config_df = pd.concat([config_df, market_maker_buy, market_maker_sell], ignore_index=True)

    return config_df


# Scenario
def create_scenario_from_config(config_json, network_path, loads_dir_path, data_dirpath=None,
                                weight_factor=1, ts_hour=4, nb_ts=None, start_date="2016-01-01",
                                plot_network=False, price_filename="basic_prices.csv",
                                ps=None, ls=None):
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

    actors = []

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
                file_dict[device['deviceType']] = device['deviceID']

        # Load
        if 'load' in file_dict:
            asset_dict['load'] = {"csv": loads_path.joinpath(file_dict['load']), "col_index": 1}

        # PV
        if 'solar' in file_dict:
            asset_dict['pv'] = {"csv": pv_path.joinpath(file_dict['solar']), "col_index": 1}

        # Prices
        asset_dict['prices'] = {"csv": price_path.joinpath(price_filename), "col_index": 1}

        actor = create_actor_from_config(actor_row['prosumerName'], asset_dict=asset_dict,
                                         start_date=start_date, nb_ts=nb_ts, ts_hour=ts_hour,
                                         ps=ps, ls=ls)

        actors.append(actor)
        print(f'{i} actor added')
        print(f'{file_dict["load"]}')

    actor_map = map_actors(config_df)
    actor_map = pn.add_actors_map(actor_map)

    if plot_network is True:
        pn.plot()
        pn.to_json()

    # Update shortest paths and the grid fee matrix
    pn.update_shortest_paths()
    pn.generate_grid_fee_matrix(weight_factor)

    return Scenario(pn, actors, actor_map)


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    config_json_path = os.path.join(dirname, 'example_config.json')
    network_path = os.path.join(dirname, 'example_network.json')
    config_path = os.path.join(dirname, 'config.txt')
    loads_dir_path = os.path.join(dirname, 'loads_dir.csv')

    data_dirpath = Path("../sample")

    parser = ArgumentParser(description='Entry point for market simulation')
    parser.add_argument('config', nargs='?', default=config_path, help='configuration file')
    args = parser.parse_args()

    cfg = Config(args.config)
    cfg.path = Path('../') / cfg.path

    sc = create_scenario_from_config(config_json_path, network_path, data_dirpath=data_dirpath,
                                     nb_ts=cfg.nb_ts, loads_dir_path=loads_dir_path, ps=1)
    sc.save(cfg.path, cfg.data_format)
    update_actor_json(cfg.path)

    if cfg.show_plots:
        sc.power_network.plot()
        sc.plot_actor_data()

    if cfg.save_network:
        sc.power_network.to_image()
        sc.power_network.to_json()
