import json
import warnings

from networkx.readwrite import json_graph
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import simply.config as cfg
from simply import actor
from simply import power_network
from simply.util import get_all_data
from simply.market_maker import MarketMaker
from simply.actor import Actor
from simply.market import Market


class Environment:
    """
    Representation of the environment which is visible to all actors. Decouples scenario information
    from actors.
    """

    def __init__(self, buy_prices, steps_per_hour, add_actor_to_scenario, **kwargs):
        self.time_step = cfg.config.start
        self.steps_per_hour = steps_per_hour
        self.add_actor_to_scenario = add_actor_to_scenario
        # Get grid fee method of market to make grid fees accessible for actors. Will be overwritten
        # when market is added to scenario
        self.get_grid_fee = Market().get_grid_fee
        if buy_prices.size == 0:
            self.market_maker = None
            warnings.warn("Environment was created without a market maker since no buy_prices, "
                          "were provided.")
        else:
            self.market_maker = MarketMaker(environment=self, buy_prices=buy_prices, **kwargs)


class Scenario:
    """
    Representation of the world state: who is present (actors) and how everything is
     connected (power_network). RNG seed is preserved so results can be reproduced.
    """

    def __init__(self,
                 network,
                 actors,
                 map_actors,
                 buy_prices: np.array = None,
                 rng_seed=None,
                 steps_per_hour=4,
                 **kwargs):

        self.rng_seed = rng_seed if rng_seed is not None else random.getrandbits(32)
        random.seed(self.rng_seed)
        self.market = None
        self.power_network = network
        self.market_participants = list(actors)
        # maps node ids to actors
        self.map_actors = map_actors
        if buy_prices is None:
            buy_prices = np.array(())
        else:
            buy_prices = np.array(buy_prices)
        self._buy_prices = buy_prices.copy()
        self.kwargs = kwargs

        self.environment = Environment(buy_prices, steps_per_hour, self.add_participant, **kwargs)

    def add_participant(self, participant):
        is_participant = isinstance(participant, Actor) or isinstance(participant, MarketMaker)
        assert is_participant
        if actor not in self.market_participants:
            self.market_participants.append(participant)

    def create_strategies(self):
        for actor_ in self.market_participants:
            if isinstance(actor_, Actor):
                actor_.get_market_schedule()

    def market_step(self):
        for participant in self.market_participants:
            orders = participant.generate_orders()
            for order in orders:
                self.market.accept_order(order, callback=participant.receive_market_results)
        self.market.clear(reset=cfg.config.reset_market)

    def next_time_step(self):
        for participant in self.market_participants:
            participant.next_time_step()

        self.environment.time_step += 1

        for participant in self.market_participants:
            participant.create_prediction()

    def from_config(self):
        pass

    def __str__(self):
        return "Scenario(network: {}, actors: {}, map_actors: {})".format(
            self.power_network, self.market_participants, self.map_actors
        )

    def to_dict(self):
        return {
            "rng_seed": self.rng_seed,
            "power_network": {self.power_network.name: self.power_network.to_dict()},
            "actors": {a.id: a.to_dict() for a in self.market_participants},
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
            for actor_variable in self.market_participants:
                a_dict[actor_variable.id] = actor_variable.to_dict(external_data=True)
                actor_variable.save_csv(dirpath)
            dirpath.joinpath('actors.json').write_text(json.dumps(a_dict, indent=2))
        else:
            # Save config and data per actor in a single file
            for actor_variable in self.market_participants:
                dirpath.joinpath(f'actor_{actor_variable.id}.{data_format}').write_text(
                    json.dumps(actor_variable.to_dict(external_data=False), indent=2)
                )

        # save map_actors
        dirpath.joinpath('map_actors.json').write_text(json.dumps(self.map_actors, indent=2))

        self.power_network.to_image(dirpath)

    def concat_participant_data(self):
        """
        Create a list of all actor data DataFrames and concatenate them using multi-column keys
        :return: DataFrame with multi-column-index (actor-level, asset-level)
        """
        data = [a.data for a in self.market_participants]
        return pd.concat(data, keys=range(len(self.market_participants)), axis=1)

    def plot_participant_data(self):
        """
        Extracts asset data from all actors of the scenario and plots all time series per asset type
        as well as the aggregated sum per asset.
        """
        actor_data = self.concat_participant_data()
        fig, ax = plt.subplots(3, sharex=True)
        ax[0].set_title("PV")
        ax[1].set_title("Load")
        ax[2].set_title("Sum")
        pv_df = get_all_data(actor_data, "pv")
        load_df = get_all_data(actor_data, "load")
        mask = load_df.apply(lambda x: abs(x) == 2 ** 63 - 1)
        pv_df.plot(ax=ax[0], legend=False)
        load_df[~mask].plot(ax=ax[1], legend=False)
        pv_df.sum(axis=1).plot(ax=ax[2])
        load_df[~mask].sum(axis=1).plot(ax=ax[2])
        ax[2].legend(["pv", "load"])
        plt.show()

    def reset(self):
        """ Reset the scenario after a simulation is run"""
        # Remove previous actors
        self.market_participants = []
        # But add the market maker
        self.market.reset()
        self.environment.market_maker = MarketMaker(
            environment=self.environment, buy_prices=self._buy_prices.copy())
        self.environment.time_step = cfg.config.start


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

    pn = power_network.create_power_network_from_config(next(dirpath.glob('network.*')))

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

    # Give actors knowledge of the cluster they belong to
    for aj in actors:
        if aj.id in pn.node_to_cluster:
            aj.cluster = pn.node_to_cluster[aj.id]

    # read map_actors
    map_actor_text = next(dirpath.glob('map_actors.*')).read_text()
    map_actors = json.loads(map_actor_text)

    return Scenario(pn, actors, map_actors, rng_seed)


def create_random(num_nodes, num_actors, weight_factor):
    pn = power_network.create_random(num_nodes)
    # Add actor nodes at random position (leaf node) in the network
    # One network node can contain several actors (using random.choices method)

    # Update shortest paths and the grid fee matrix
    pn.update_shortest_paths()
    pn.generate_grid_fee_matrix(weight_factor)
    mm_buy_prices = np.random.random(100)
    scenario = Scenario(pn, [], None, buy_prices=mm_buy_prices)
    environment = scenario.environment
    actors = [actor.create_random("H" + str(i), environment=environment) for i in range(num_actors)]
    map_actors = pn.add_actors_random(actors)
    scenario.map_actors = map_actors
    return scenario


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
    mm_buy_prices = np.random.random(100)

    return Scenario(pn, actors, map_actors, mm_buy_prices)


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

    return Scenario(pn, actors, map_actors, steps_per_hour=ts_hour)
