import json
from networkx.readwrite import json_graph
import pandas as pd
import random
import matplotlib.pyplot as plt

from simply import actor
from simply import power_network
from simply.util import get_all_data


class Scenario:
    """
    Representation of the world state: who is present (actors) and how everything is
     connected (power_network). RNG seed is preserved so results can be reproduced.
    """

    def __init__(self, network, actors, map_actors, rng_seed=None):
        self.rng_seed = rng_seed if rng_seed is not None else random.getrandbits(32)
        random.seed(self.rng_seed)

        self.power_network = network
        self.actors = list(actors)
        # maps node ids to actors
        self.map_actors = map_actors

    def from_config(self):
        pass

    def __str__(self):
        return "Scenario(network: {}, actors: {}, map_actors: {})".format(
            self.power_network, self.actors, self.map_actors
        )

    def to_dict(self):
        return {
            "rng_seed": self.rng_seed,
            "power_network": {self.power_network.name: self.power_network.to_dict()},
            "actors": {a.id: a.to_dict() for a in self.actors},
            "map_actors": self.map_actors,
        }

    def save(self, dirpath, data_format):
        """
        Save scenario files to directory

        dirpath: Path object
        """
        # create target directory
        dirpath.mkdir(parents=True, exist_ok=True)

        # save meta information
        dirpath.joinpath('_meta.inf').write_text(json.dumps({"rng_seed": self.rng_seed}, indent=2))

        # save power network
        dirpath.joinpath('network.json').write_text(
            json.dumps(
                {self.power_network.name: self.power_network.to_dict()},
                indent=2,
            )
        )

        # save actors
        if data_format == "csv":
            # Save data in separate csv file and all actors in one config file
            a_dict = {}
            for actor_variable in self.actors:
                a_dict[actor_variable.id] = actor_variable.to_dict(external_data=True)
                actor_variable.save_csv(dirpath)
            dirpath.joinpath('actors.json').write_text(json.dumps(a_dict, indent=2))
        else:
            # Save config and data per actor in a single file
            for actor_variable in self.actors:
                dirpath.joinpath(f'actor_{actor_variable.id}.{data_format}').write_text(
                    json.dumps(actor_variable.to_dict(external_data=False), indent=2)
                )

        # save map_actors
        dirpath.joinpath('map_actors.json').write_text(json.dumps(self.map_actors, indent=2))

    def concat_actor_data(self):
        """
        Create a list of all actor data DataFrames and concatenate them using multi-column keys
        :return: DataFrame with multi-column-index (actor-level, asset-level)
        """
        data = [a.data for a in self.actors]
        return pd.concat(data, keys=range(len(self.actors)), axis=1)

    def plot_actor_data(self):
        """
        Extracts asset data from all actors of the scenario and plots all time series per asset type
        as well as the aggregated sum per asset.
        """
        actor_data = self.concat_actor_data()
        fig, ax = plt.subplots(3)
        ax[0].set_title("PV")
        ax[1].set_title("Load")
        ax[2].set_title("Sum")
        get_all_data(actor_data, "pv").plot(ax=ax[0], legend=False)
        get_all_data(actor_data, "load").plot(ax=ax[1], legend=False)
        get_all_data(actor_data, "pv").sum(axis=1).plot(ax=ax[2])
        get_all_data(actor_data, "load").sum(axis=1).plot(ax=ax[2])
        ax[2].legend(["pv", "load"])
        plt.show()


def from_dict(scenario_dict):
    pn_name, pn_dict = scenario_dict["power_network"].popitem()
    assert len(scenario_dict["power_network"]) == 0, "Multiple power networks in scenario"
    network = json_graph.node_link_graph(pn_dict,
                                         directed=pn_dict.get("directed", False),
                                         multigraph=pn_dict.get("multigraph", False))
    pn = power_network.PowerNetwork(pn_name, network)

    actors = [
        actor.Actor(actor_id, pd.read_json(ai["df"]), ai["ls"], ai["ps"], ai["pm"])
        for actor_id, ai in scenario_dict["actors"].items()]

    return Scenario(pn, actors, scenario_dict["map_actors"], scenario_dict["rng_seed"])


def load(dirpath, data_format):
    """
    Create scenario from files that were generated by Scenario.save()

    dirpath: Path object
    """

    # read meta info
    meta_text = dirpath.joinpath('_meta.inf').read_text()
    meta = json.loads(meta_text)
    rng_seed = meta.get("rng_seed", None)

    # read power network
    network_text = next(dirpath.glob('network.*')).read_text()
    network_json = json.loads(network_text)
    network_name = list(network_json.keys())[0]
    network_json = list(network_json.values())[0]
    network = json_graph.node_link_graph(network_json,
                                         directed=network_json.get("directed", False),
                                         multigraph=network_json.get("multigraph", False))
    pn = power_network.PowerNetwork(network_name, network)

    # read actors

    actors = []
    if data_format == "csv":
        actors_file = next(dirpath.glob("actors.*"))
        at = actors_file.read_text()
        actors_j = json.loads(at)
        for aj in actors_j.values():
            ai = [aj["id"], pd.read_csv(dirpath / aj["csv"]), aj["csv"], aj["ls"], aj["ps"],
                  aj["pm"]]
            actors.append(actor.Actor(*ai))
    else:
        actor_files = dirpath.glob(f"actor_*.{data_format}")
        for f in sorted(actor_files):
            at = f.read_text()
            aj = json.loads(at)
            ai = [aj["id"], pd.read_json(aj["df"]), aj["csv"], aj["ls"], aj["ps"], aj["pm"]]
            actors.append(actor.Actor(*ai))

    for aj in actors:
        if aj.id in pn.node_to_cluster:
            aj.grid_id = pn.node_to_cluster[aj.id]

    # read map_actors
    map_actor_text = next(dirpath.glob('map_actors.*')).read_text()
    map_actors = json.loads(map_actor_text)

    return Scenario(pn, actors, map_actors, rng_seed)


def create_random(num_nodes, num_actors, weight_factor):
    pn = power_network.create_random(num_nodes)
    actors = [actor.create_random("H" + str(i)) for i in range(num_actors)]

    # Add actor nodes at random position (leaf node) in the network
    # One network node can contain several actors (using random.choices method)
    map_actors = pn.add_actors_random(actors)
    # Update shortest paths and the grid fee matrix
    pn.update_shortest_paths()
    pn.generate_grid_fee_matrix(weight_factor)

    return Scenario(pn, actors, map_actors)


def create_random2(num_nodes, num_actors):
    assert num_actors < num_nodes
    # num_actors has to be much smaller than num_nodes
    pn = power_network.create_random(num_nodes)
    actors = [actor.create_random("H" + str(i)) for i in range(num_actors)]

    # Give actors a random position in the network
    actor_nodes = random.sample(pn.leaf_nodes, num_actors)
    map_actors = {actor.id: node_id for actor, node_id in zip(actors, actor_nodes)}

    # TODO tbd if actors are already part of topology ore create additional nodes
    # pn.add_actors_map(map_actors)

    return Scenario(pn, actors, map_actors)


def create_scenario_from_csv(dirpath, num_nodes, num_actors, weight_factor, ts_hour=4, nb_ts=None):
    """
    Load csv files from path and randomly select num_actors to be randomly

    :param dirpath: Path object
    :param num_nodes: number of nodes in the network
    :param num_actors: number of actors in the network
    :param weight_factor: weight factor used to derive grid fees
    :param ts_hour: number of time slot of equal length within one hour
    :param nb_ts: number of time slots to be generated
    """
    # Create random nodes for power network
    pn = power_network.create_random(num_nodes)

    # Read all filenames from given directory
    filenames = dirpath.glob("*.csv")
    # Choose a random sample of files to read
    filenames = random.sample(list(filenames), num_actors)

    # Assign csv file to actor and save dictionary
    household_type = {}
    # create initial list of actors
    actors = []

    # iterate over list of files to be read to update actors
    for i, filename in enumerate(filenames):
        # save actor_id and data description in list
        household_type.update({i: filename.stem})
        print('actor_id: {} - household: {}'.format(i, household_type[i]))
        # read file
        a = actor.create_from_csv(
            "H_" + str(i),
            asset_dict={
                "load": {"csv": filename, "col_index": 1},
                "pv": {}
            },
            start_date="2021-01-01",
            nb_ts=nb_ts,
            ts_hour=ts_hour
        )

        actors.append(a)

    map_actors = pn.add_actors_random(actors)

    # Update shortest paths and the grid fee matrix
    pn.update_shortest_paths()
    pn.generate_grid_fee_matrix(weight_factor)

    return Scenario(pn, actors, map_actors)
