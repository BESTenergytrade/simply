import datetime
import random
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from networkx.readwrite import json_graph

from simply.actor import Actor
from simply.scenario import Scenario
from simply.power_network import PowerNetwork
from simply.config import Config


def end_date_from_ts(start_date="2016-01-01", nb_ts=None, ts_hour=1):
    start_date = pd.to_datetime(start_date)
    # Rename column and insert data based on dictionary
    ts_minutes = (nb_ts - 1) * (60 / ts_hour)
    time_change = datetime.timedelta(minutes=ts_minutes)
    end_date = start_date + time_change
    return start_date, end_date


def basic_strategy(df):
    df["schedule"] = df["pv"] - df["load"]
    return df


def dummy_strategy(df, ts_hour, nb_ts, max_price=0.3, net_price_factor=0.7, pv_prob=0.4):
    # Random scale factor generation, load and price time series in boundaries
    peak = {
        "load": random.uniform(0.8, 1.3) / ts_hour,
        "pv": random.uniform(1, 7) / ts_hour
    }
    # Probability of an actor to possess a PV, here 40%
    peak["pv"] = random.choices([0, peak["pv"]], [1 - pv_prob, pv_prob], k=1)

    # Dummy-Strategy:
    # Predefined energy management, energy volume and price for trades due to no flexibility
    df["schedule"] = peak["pv"] * df["pv"] - peak["load"] * df["load"]
    df["prices"] = np.random.rand(nb_ts, 1)
    df["prices"] *= max_price
    # Adapt order price by a factor to compensate net pricing of ask orders
    # (i.e. positive power) Bids however include network charges
    net_price_factor = 0.7
    df["prices"] = df.apply(
        lambda slot: slot["prices"] - (slot["schedule"] > 0) * net_price_factor * slot["prices"],
        axis=1
    )
    return df


# Helper function to build power network from community config json
def map_actors(config_df):
    map = {}
    for i in config_df.index:
        map[config_df["prosumerName"][i]] = config_df["gridLocation"][i]
    return map


# Power Network
def create_power_network_from_config(network_path, weight_factor=1):
    with open(network_path) as user_file:
        file_contents = user_file.read()

    network_json = json.loads(file_contents)

    network_name = list(network_json.keys())[0]
    network_json = list(network_json.values())[0]
    network = json_graph.node_link_graph(network_json,
                                         directed=network_json.get("directed", False),
                                         multigraph=network_json.get("multigraph", False))
    return PowerNetwork(network_name, network, weight_factor)


# Actor
def create_actor_from_config(actor_id, asset_dict={}, start_date="2016-01-01", nb_ts=None,
                             ts_hour=1, cols=["load", "pv", "schedule", "prices"], pv_prob=0.4):
    df = pd.DataFrame([], columns=cols)
    start_date, end_date = end_date_from_ts(start_date, nb_ts, ts_hour)

    # Read csv files for each asset
    csv_peak = {}
    for col, csv_dict in asset_dict.items():
        # if csv_dict is empty
        if not csv_dict:
            continue
        csv_df = pd.read_csv(
            csv_dict["csv"],
            sep=',',
            parse_dates=['Time'],
            dayfirst=True,
            index_col=['Time']
        )

        df.loc[:, col] = csv_df[start_date:end_date]
        # Save peak value and normalize time series
        csv_peak[col] = df[col].max()

    # df = dummy_strategy(df, ts_hour=ts_hour, nb_ts=nb_ts, pv_prob=pv_prob)
    df = basic_strategy(df)

    return Actor(actor_id, df)


def read_config_json(config_json):
    config_df = pd.read_json(config_json)
    if 'market_maker' in list(config_df['prosumerType']):
        # config_df.drop('market_maker', axis=1)

        config_df = config_df[config_df.prosumerType != 'market_maker']
        config_df = config_df.append({'prosumerName': 'market_maker_sell',
                                      'prosumerType': 'market_maker_sell', 'gridLocation':
                                          'market_maker'}, ignore_index=True)
        config_df = config_df.append({'prosumerName': 'market_maker_buy',
                                      'prosumerType': 'market_maker_buy',
                                      'gridLocation': 'market_maker'}, ignore_index=True)

    return config_df


# Scenario
def create_scenario_from_config(config_json, network_path, loads_dir_path, data_dirpath=None,
                                weight_factor=1, ts_hour=4, nb_ts=None, start_date="2016-01-01",
                                plot_network=False, price_filename="basic_prices.csv", pv_prob=1,
                                normalised=False):
    loads_path = data_dirpath.joinpath("load")
    pv_path = data_dirpath.joinpath("pv")
    price_path = data_dirpath.joinpath("price")

    # Parse json
    config_df = read_config_json(config_json)
    # Create nodes for power network
    pn = create_power_network_from_config(network_path, weight_factor)

    if plot_network is True:
        pn.plot()

    actors = []
    file_dict = {}
    asset_dict = {}

    for i, actor_row in config_df.iterrows():
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
                                         pv_prob=pv_prob)

        actors.append(actor)
        print(f'{i} actor added')
        print(f'{file_dict["load"]}')

    actor_map = map_actors(config_df)
    actor_map = pn.add_actors_map(actor_map)

    if plot_network is True:
        pn.plot()

    # Update shortest paths and the grid fee matrix
    pn.update_shortest_paths()
    pn.generate_grid_fee_matrix(weight_factor)
    pn.to_json()

    return Scenario(pn, actors, actor_map)


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    config_json_path = os.path.join(dirname, 'example_config.json')
    network_path = os.path.join(dirname, 'example_network.json')
    config_path = os.path.join(dirname, 'config.txt')
    loads_dir_path = os.path.join(dirname, 'loads_dir.csv')

    data_dirpath = Path("../sample")
    sc_path = Path("../scenarios/default")

    parser = ArgumentParser(description='Entry point for market simulation')
    parser.add_argument('config', nargs='?', default=config_path, help='configuration file')
    args = parser.parse_args()

    cfg = Config(args.config)
    cfg.nb_ts = 3 * 96

    sc = create_scenario_from_config(config_json_path, network_path, data_dirpath=data_dirpath,
                                     nb_ts=cfg.nb_ts, loads_dir_path=loads_dir_path)
    sc.save(sc_path, cfg.data_format)

    if cfg.show_plots:
        sc.power_network.plot()
        sc.plot_actor_data()

    sc.power_network.to_image()
    sc.power_network.to_json()
