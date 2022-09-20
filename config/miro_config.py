from argparse import ArgumentParser
import json
import random
from pathlib import Path
import pandas as pd
import numpy as np
from networkx.readwrite import json_graph

from simply.actor import Actor
from simply.scenario import Scenario
from simply.power_network import PowerNetwork
from simply.config import Config
from simply.util import daily, gaussian_pv

"""Key functionality:
1) Allow user to select how many (and what type of agents) to include
2) """


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
def create_actor_from_config(actor_id, asset_dict={}, start_date="2021-01-01", nb_ts=None, ts_hour=1):
    """
    Create actor instance with random asset time series and random scaling factors. Replace

    :param str actor_id: unique actor identifier
    :param Dict asset_dict: nested dictionary specifying 'csv' filename and column ('col_index')
        per Actor asset
    :param str start_date: Start date "YYYY-MM-DD" of the DataFrameIndex for the generated actor's
        asset time series
    :param int nb_ts: number of time slots that should be generated, derived from csv if None
    :param ts_hour: number of time slots per hour, e.g. 4 results in 15min time slots
    :return: generated Actor object
    :rtype: Actor
    """
    # Random scale factor generation, load and price time series in boundaries
    ls = random.uniform(0.8, 1.3) / ts_hour
    ps = random.uniform(1, 7) / ts_hour
    # Probability of an actor to possess a PV, here 40%
    pv_prob = 0.4
    ps = random.choices([0, ps], [1 - pv_prob, pv_prob], k=1)

    # Initialize DataFrame
    cols = ["load", "pv", "schedule", "prices"]
    df = pd.DataFrame([], columns=cols)

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
            dayfirst=True
        )
        # Rename column and insert data based on dictionary
        df.loc[:, col] = csv_df.iloc[:nb_ts, csv_dict["col_index"]]
        # Save peak value and normalize time series
        csv_peak[col] = df[col].max()
        df[col] = df[col] / df[col].max()

    # Set index
    if "index" not in df.columns:
        df["index"] = pd.date_range(
            start=start_date,
            freq="{}min".format(int(60 / ts_hour)),
            periods=nb_ts
        )
    df = df.set_index("index")

    # If pv asset key is present but dictionary does not contain a filename
    if "pv" in asset_dict.keys() and not asset_dict["pv"].get("filename"):
        # Initialize PV with random noise
        df["pv"] = np.random.rand(nb_ts, 1)
        # Multiply random generation signal with gaussian/PV-like characteristic per day

        for day in daily(df, 24 * ts_hour):
            day["pv"] *= gaussian_pv(ts_hour, 3)

    # Dummy-Strategy:
    # Predefined energy management, energy volume and price for trades due to no flexibility
    df["schedule"] = ps * df["pv"] - ls * df["load"]
    max_price = 0.3
    df["prices"] = np.random.rand(nb_ts, 1)
    df["prices"] *= max_price
    # Adapt order price by a factor to compensate net pricing of ask orders
    # (i.e. positive power) Bids however include network charges
    net_price_factor = 0.7
    df["prices"] = df.apply(
        lambda slot: slot["prices"] - (slot["schedule"] > 0) * net_price_factor
                     * slot["prices"], axis=1
    )

    return Actor(actor_id, df, ls=ls, ps=ps)


# Scenario
def create_scenario_from_config(config_json, network_path, data_dirpath=None, weight_factor=1,
                                ts_hour=4, nb_ts=None, start_date="2021-01-01"):
    # Parse json
    config_df = pd.read_json(config_json)

    # for
    #
    #
    #

    # Create nodes for power network
    pn = create_power_network_from_config(network_path, weight_factor)
    pn.plot()

    num_actors = len(config_df.index)
    # Read all filenames from given directory

    filenames = data_dirpath.glob("*.csv")
    # Choose a random sample of files to read
    # x = list(filenames)
    filenames = random.sample(list(filenames), num_actors)

    # Assign csv file to actor and save dictionary
    household_type = {}
    # create initial list of actors
    actors = []

    # iterate over list of files to be read to update actors
    for i, filename in enumerate(filenames):
        # save actor_id and data description in list
        household_type.update({i: filename.stem})
        print(f'actor_id: {i} - household: {household_type[i]}')
        # read file
        a = create_actor_from_config(
            "H_" + str(i),
            asset_dict={
                "load": {"csv": filename, "col_index": 1},
                # "pv": {}
            },
            start_date=start_date,
            nb_ts=nb_ts,
            ts_hour=ts_hour
        )

        actors.append(a)

    map_actors = pn.add_actors_random(actors)

    # Update shortest paths and the grid fee matrix
    pn.update_shortest_paths()
    pn.generate_grid_fee_matrix(weight_factor)

    return Scenario(pn, actors, map_actors)


if __name__ == "__main__":
    # """Check that the argument parsing below is necessary"""
    parser = ArgumentParser(description='Entry point for market simulation')
    parser.add_argument('config', nargs='?', default="", help='configuration file')
    args = parser.parse_args()
    cfg = Config(args.config)

    # Include the absolute paths here:
    config_json_path = '/Users/emilmargrain/Documents/GitHub/simply/config/community_config.json'
    network_path = '/Users/emilmargrain/Documents/GitHub/simply/config/network.json'

    data_path = Path("../sample", "households_sample")

    sc = create_scenario_from_config(config_json_path, network_path, data_path, nb_ts=3*96)
