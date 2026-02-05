import os
import json
import warnings
from typing import Sized, Iterable
from networkx.readwrite import json_graph
import pandas as pd
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

import simply.config as cfg
from simply import actor, market_maker
from simply import power_network
# from simply.battery import Battery
from simply.util import get_all_data, timeit
from simply.market_maker import MarketMaker
from simply.actor import Actor
from simply.market import Market
from simply.util import run_obj_method
from simply.defaults import MARKETID

try:
    matplotlib.use('TkAgg')
except ImportError as e:
    print(e)
debug_actor = None  # 'residential_3'


class Environment:
    """Representation of the environment which is visible to all actors. Decouples scenario
    information from actors.

    :param iterable buy_prices: iterable of prices the market maker would buy energy for
    :param int steps_per_hour: amount of simulation steps per hour
    :param function add_actor_to_scenario: Function which adds the actor to
        py:attr:`~simply.scenario.actors`
    :param dict kwargs: kwargs for MarketMaker generation

    Members:

    time_step : int
        current time_step of the environment
    steps_per_hour : int
        amount of simulation steps per hour
    add_actor_to_scenario : function
        Function which adds the actor to py:attr:`~simply.scenario.actors`
    time_range : pd.DatetimeIndex / int
        index that is used for all outputs and data indexing;
         if it is not provided an int range is used based on the config
    get_grid_fee : method
        getter function of grid_fee of the Market
    market_maker : py:class:`~simply.market_maker.MarketMaker`
        market_maker in this environment (deprecated)
    market_makers : list(py:class:`~simply.market_maker.MarketMaker`)
        list of market_maker objects of this environment
    """

    def __init__(self, steps_per_hour, add_actor_to_scenario, time_range=None, **kwargs):
        self.time_step = cfg.config.start
        self.steps_per_hour = steps_per_hour if steps_per_hour is not None else cfg.config.ts_per_hour
        if time_range is None:
            self.time_range = range(cfg.config.start + cfg.config.nb_ts + 1)
        elif isinstance(time_range, pd.DatetimeIndex):
            # Check for correct datetime frequency
            if time_range.freq is None:
                time_range.freq = pd.infer_freq(time_range)
            print(f"Found date time index starting at {time_range[0]} "
                  f"with freq {time_range.freq}")
            if time_range.freq != "{}T".format(60 / self.steps_per_hour):
                warnings.warn(f"Time Index of data frequency {str(time_range.freq)} "
                              f"does not match the configured steps_per_hour: {self.steps_per_hour}")
            self.time_range = time_range
        else:
            self.time_range = time_range
        self.add_actor_to_scenario = add_actor_to_scenario
        # Get grid fee method of market to make grid fees accessible for actors. Will be overwritten
        # when market is added to scenario
        self.get_grid_fee = None  # is instance of Market().get_grid_fee
        self.get_grid_fee_dict = {}
        self.market_maker: MarketMaker = None  # TODO: remove legacy
        self.market_makers = {}


def is_scenario_participant(obj):
    if isinstance(obj, Actor):
        return True
    if isinstance(obj, MarketMaker):
        return True
    return False


class Scenario:
    """
    Representation of the world state: who is present (actors) and how everything is
     connected (power_network). RNG (random number generator) seed is preserved so
     results can be reproduced.
    """

    def __init__(self, network, map_actors=None, buy_prices: np.array = None, rng_seed=None,
                 steps_per_hour=None, time_range=None, **kwargs):

        self.rng_seed = rng_seed if rng_seed is not None else random.getrandbits(32)
        random.seed(self.rng_seed)
        self._market = None
        self.market_dict = {}
        self.power_network: power_network.PowerNetwork = network
        self.market_participants = list()
        # maps node ids to actors
        if map_actors is None:
            map_actors = {}
        self.map_actors: dict = map_actors

        self.kwargs = kwargs
        self.environment = Environment(steps_per_hour, self.add_participant, time_range, **kwargs)
        if buy_prices is None:
            buy_prices = np.array(())
        else:
            buy_prices = np.array(buy_prices)
            self.add_market_maker(buy_prices, **kwargs)
        if "market" in kwargs.keys():
            self.set_market(kwargs["market"])

    def add_market_maker(self, buy_prices: Sized, **kwargs):
        if len(buy_prices) == 0:
            warnings.warn("Environment was created without a market maker since no buy_prices, "
                          "were provided.")
        else:
            # Create the Market maker. Since the environment is passed the MarketMaker automatically
            # adds itself to the environment and also to the scenario participants
            MarketMaker(buy_prices=buy_prices, environment=self.environment, **kwargs)

    def get_market(self):
        return self._market

    def set_market(self, market):
        assert isinstance(market, Market), "Scenario.market can only be changed to type 'Market'"
        self._market = market
        self._market.t_step = self.environment.time_step
        self.environment.get_grid_fee = self._market.get_grid_fee

    # creating a property object. This way changing markets also leads to changes in grid fee
    market = property(get_market, set_market)

    def add_participants(self, participants: Iterable, map_actors=None, add_to_network=False):
        """Add participants to the scenario.

        If added to the network nodes, this is either done using the provided mapping dictionary or
        randomly.

        :param participants: participants to be added
        :type participants: simply.actor.Actor or simply.scenario.MarketMaker
        :param map_actors: dictionary with actor_ids as key and node_id as value
        :param add_to_network: should the actors be added to the power network
        :type add_to_network: bool
        :return:
        """
        actors = list(filter(lambda x: isinstance(x, Actor), participants))

        if map_actors is None:
            if add_to_network:
                # Add the only actors randomly to the power network

                map_actors = self.power_network.add_actors_random(actors)
            else:
                map_actors = {}
        else:
            # Make sure there are as many unique actor_ids as actors
            actor_ids = [actor_.id for actor_ in actors]
            assert len(actors) == len(set(actor_ids))

            # Make sure every actor is found exactly once in map_actors and map_actors does not
            # contain not used actors
            assert set(actor_ids) == set(map_actors.keys())

            if add_to_network:
                self.power_network.add_actors_map(map_actors)

        # Only update the node mapping of the provided actors
        self.map_actors.update(map_actors)

        for participant in participants:
            self._add_participant(participant)

        mm_list = [x for x in self.market_participants if isinstance(x, MarketMaker)]
        if mm_list != 0:
            print(f" + Added {len(mm_list)} MarketMakers to the Scenario.")
        print(f" + Added {len(actors)} Actors to the Scenario.")

    def add_participant(self, participant, map_node=None, add_to_network=False):
        self._add_participant(participant)
        map_actors = {}
        if map_node is None:
            if add_to_network:
                map_actors = self.power_network.add_actors_random([participant])
        else:
            map_actors = {participant.id: map_node}
            if add_to_network:
                _ = self.power_network.add_actors_map(map_actors)
        self.map_actors.update(map_actors)

    def _add_participant(self, participant):
        assert is_scenario_participant(participant)
        if participant not in self.market_participants:
            if isinstance(participant, MarketMaker):
                self.environment.market_makers[participant.id] = participant
            self.market_participants.append(participant)
        else:
            warnings.warn(f"Participant {participant} is already part of the scenario, and was "
                          f"not added again.")
        participant.environment = self.environment
        participant.create_prediction()

    @timeit
    def create_strategies(self, max_workers=None, update_step=1):
        # only actors create strategies (in parallel)
        actors = [p for p in self.market_participants if isinstance(p, Actor)]
        if not actors:
            return

        if self.environment.time_step % update_step != 0:
            for a in actors:
                a.shift_market_schedule()
            return

        if max_workers is None:
            max_workers = os.cpu_count() or 1

        # Parallel: create schedule per actor and update object in main process
        # - cbc is running as external program which is why ThreadPoolExecutor is sufficient
        # (the use of ProcessPoolExecutor even leads to longer execution time due to pickling
        #  overhead)
        # if process pool is used the values have to be updated due to separate memory
        process_execution = False
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(run_obj_method, a, "get_market_schedule"): a for a in actors}
            if process_execution:
                raise NotImplementedError
                for fut in as_completed(futs):
                    actor = futs[fut]
                    market_schedule = fut.result()
                    # writing back the result due to mutability
                    # (only necessary for Process Execution)
                    # TODO check for missing updates other than market_schedule
                    actor.market_schedule = market_schedule

    @timeit
    def create_strategies_sequential(self, update_step=1):
        # sequential execution of market_schedule creation
        actors = [p for p in self.market_participants if isinstance(p, Actor)]
        if self.environment.time_step % update_step != 0:
            for a in actors:
                a.shift_market_schedule()
            return
        else:
            for a in actors:
                a.get_market_schedule()

    def add_market(self, market):
        self.market = market
        self.sync_market_time()

    def add_to_market_dict(self, market, market_name=MARKETID):
        assert isinstance(market, Market), "Only Instances of class 'Market' can be added to Scenario.market_dict"
        if market_name in self.market_dict.keys():
            warnings.warn(f"Market named {market_name} already exists.")
        self.market_dict[market_name] = market
        self.market_dict[market_name].t_step = self.environment.time_step
        self.market_dict[market_name].t_step = self.environment.time_range[self.environment.time_step]
        self.market_dict[market_name].step = self.environment.time_step
        self.environment.get_grid_fee_dict[market_name] = self.market_dict[market_name].get_grid_fee

    def sync_market_time(self):
        for market in self.market_dict.values():
            market.t_step = self.environment.time_range[self.environment.time_step]
            market.step = self.environment.time_step

    @timeit
    def market_step(self):
        for participant in self.market_participants:
            orders = participant.generate_orders()
            for order in orders:
                if isinstance(participant, Actor):
                    self.market_dict[participant.assigned_market].accept_order(
                        order, callback=participant.receive_market_results)
                else:  # MarketMaker
                    for market in participant.assigned_market:
                        self.market_dict[market].accept_order(order, callback=participant.receive_market_results)
        if debug_actor:
            print([order for order in orders if "MarketMaker" != order.actor_id])
            for m in self.market_dict.values():
                print(m.orders)
        for market in self.market_dict.values():
            market.clear(reset=cfg.config.reset_market)
        if debug_actor:
            for mark in self.market_dict.values():
                print([m for ma in mark.matches for m in ma if
                       m["time"] == self.environment.time_step])
                print([m for matches in mark.matches for m in matches
                       if m["time"] == self.environment.time_step
                       and (m["bid_actor"] == debug_actor or m["ask_actor"] == debug_actor)])

    def next_time_step(self):
        for participant in self.market_participants:
            participant.prepare_next_time_step()
        if isinstance(self.environment.time_step, int):
            self.environment.time_step += 1
        else:
            raise TypeError("Expect time_step to be of type: int")
        for participant in self.market_participants:
            participant.create_prediction()
        self.sync_market_time()

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
                indent=2, default=serialize_int64
            )
        )

        # save actors
        if data_format == "csv":
            # Save data in separate csv file and all actors in one config file
            a_dict = {}
            m_dict = {}
            for participant in self.market_participants:
                if isinstance(participant, Actor):
                    a_dict[participant.id] = participant.to_dict(external_data=True)
                if isinstance(participant, MarketMaker):
                    m_dict[participant.id] = participant.to_dict(external_data=True)
                participant.save_csv(dirpath)
            dirpath.joinpath('actors.json').write_text(
                json.dumps({"actors": a_dict, "marketMakers": m_dict}, indent=2, default=serialize_int64))
        else:
            # Save config and data per actor in a single file
            for participant in self.market_participants:
                dirpath.joinpath(f'actor_{participant.id}.{data_format}').write_text(
                    json.dumps(participant.to_dict(external_data=False), indent=2,
                               default=serialize_int64)
                )

        # save map_actors
        dirpath.joinpath('map_actors.json').write_text(json.dumps(self.map_actors, indent=2))

        self.power_network.to_image(dirpath)

    def save_additional_results(self, dirpath):
        for a in list(filter(lambda x: isinstance(x, Actor), self.market_participants)):
            a.save_actor_result(dirpath / f"actor_{a.id}.csv")
        print("Additional actor results saved.")

    def track_actor_schedule(self, dirpath, actor_id):
        for a in list(filter(lambda x: isinstance(x, Actor), self.market_participants)):
            if a.id == actor_id:
                a.save_actor_schedule(dirpath / f"actor_{a.id}_schedule.csv")
                print(f"Tracking actor {actor_id} schedule (saved)")
                return

    def concat_actors_data(self):
        """
        Create a list of all actor data DataFrames and concatenate them using multi-column keys
        :return: DataFrame with multi-column-index (actor-level, asset-level)
        """
        actors = list(filter(lambda x: isinstance(x, Actor), self.market_participants))
        data = [a.data for a in actors]

        return pd.concat(data, keys=[actor_.id for actor_ in actors], axis=1)

    def plot_participant_data(self):
        """
        Extracts asset data from all actors of the scenario and plots all time series per asset type
        as well as the aggregated sum per asset.
        """
        actor_data = self.concat_actors_data()
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

    def plot_prices(self):
        if self.environment.market_maker is not None:
            fig, ax = plt.subplots(1, sharex=True)
            ax = [ax]
            ax[0].plot([p + cfg.config.default_grid_fee for p in
                        self.environment.market_maker.all_sell_prices])
            ax[0].plot(self.environment.market_maker.all_buy_prices)
            plt.show()

    def reset(self):
        """ Reset the scenario after a simulation is run"""
        # Reset the time step for the environment and all markets
        self.environment.time_step = cfg.config.start
        for m in self.market_dict.values():
            m.t_step = self.environment.time_step
            m.reset()

        # Remove previous participants
        self.market_participants = []

        # Store the old market maker
        if self.environment.market_maker is not None:
            raise NotImplementedError()
            # TODO remove legacy
            market_maker = self.environment.market_maker
            market_maker.reset()
            # But add the market maker again
            self.add_participant(market_maker)
        # adaptation for multi market makers
        market_makers = self.environment.market_makers
        for mm in market_makers.values():
            mm.reset()
            self.add_participant(mm)


def serialize_int64(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError("Type %s is not serializable" % type(obj))


def from_dict(scenario_dict):
    pn_name, pn_dict = scenario_dict["power_network"].popitem()
    assert len(scenario_dict["power_network"]) == 0, "Multiple power networks in scenario"
    network = json_graph.node_link_graph(pn_dict,
                                         directed=pn_dict.get("directed", False),
                                         multigraph=pn_dict.get("multigraph", False))
    pn = power_network.PowerNetwork(pn_name, network)

    scen = Scenario(pn, scenario_dict["map_actors"], scenario_dict["rng_seed"])
    for actor_id, ai in scenario_dict["actors"].items():
        actor_ = actor.Actor(actor_id, pd.read_json(ai["df"]), ai["ls"], ai["ps"], ai["pm"])
        scen.add_participant(actor_)
    return scen


def load(dirpath, data_format):
    """
    Create scenario from files that were generated by Scenario.save()

    :param dirpath: Path object
    :param data_format: File ending of actor data e.g. `csv`
    """

    # read meta info
    meta_text = dirpath.joinpath('_meta.inf').read_text()
    meta = json.loads(meta_text)
    rng_seed = meta.get("rng_seed", None)

    pn = power_network.create_power_network_from_config(
        next(dirpath.glob('network.*')), weight_factor=cfg.config.weight_factor)

    markets_json = dirpath / "markets.json"
    if markets_json.is_file():
        existing_markets = []
        with open(markets_json) as f:
            for mc in json.load(f):
                existing_markets.append(mc["market_name"])
    else:
        existing_markets = [MARKETID]
    assert len(existing_markets) != 0, "At least one market has to be defined"

    # read actors
    participants = []
    time_range = None
    market_list = []
    mm_markets = {}  # per maket maker: aqssigned_market
    from datetime import datetime
    if data_format == "csv":
        actors_file = next(dirpath.glob("actors.*"))
        at = actors_file.read_text()
        actors_j = json.loads(at)
        for aj in actors_j["actors"].values():
            if aj["assignedMarket"] in existing_markets:
                # market.json is not empty
                aj["df"] = pd.read_csv(dirpath / aj["csv"], parse_dates=['Time'], dayfirst=False,
                                       index_col='Time')
                assert datetime.strptime(cfg.config.start_date, "%Y-%m-%d") in aj["df"].index
                time_range = aj["df"].index
                participant = actor.Actor(**aj)
                participants.append(participant)
        for mj in actors_j["marketMakers"].values():
            # filter the assigned markets with the available markets i.e. from market.json
            filtered_market = list(set(existing_markets) & set(mj["assignedMarket"]))
            if len(filtered_market) > 0:
                mj["assignedMarket"] = filtered_market
                m_df = pd.read_csv(dirpath / mj["csv"])
                mj["buy_prices"] = list(m_df["all_buy_prices"])
                if "all_sell_prices" in m_df.columns:
                    mj["sell_prices"] = m_df["all_sell_prices"]
                participant = market_maker.MarketMaker(**mj)
                market_list += participant.assigned_market
                mm_markets[participant.id] = participant.assigned_market
                participants.append(participant)
        # check that every market has only one market maker assigned
        assert len(set(market_list)) == len(market_list), 'markets with more than one assigned market maker'
        for p in participants:
            # check for every Actor, that its assigned market_maker actually trades in its assigned market
            if isinstance(p, Actor):
                assert p.assigned_market in mm_markets[p.assigned_mm], (
                    f"{p.id} is assigned to {p.assigned_market} and {p.assigned_mm}. "
                    f"But this market maker does not trade on this market.")
    else:
        actor_files = dirpath.glob(f"actor_*.{data_format}")
        for f in sorted(actor_files):
            at = f.read_text()
            aj = json.loads(at)
            if aj["id"] == market_maker.MARKETMAKERID:
                participant = market_maker.MarketMaker(**aj)
            else:
                aj["df"] = pd.read_json(aj["df"])
                aj["df"] = aj["df"].set_index("Time")
                aj["df"].index = pd.to_datetime(aj["df"].index)
                assert datetime.strptime(cfg.config.start_date, "%Y-%m-%d") in aj["df"].index
                time_range = aj["df"].index
                participant = actor.Actor(**aj)
            participants.append(participant)

    # Give actors knowledge of the cluster they belong to
    for aj in participants:
        if aj.id in pn.node_to_cluster:
            aj.cluster = pn.node_to_cluster[aj.id]

    # read map_actors
    map_actor_text = next(dirpath.glob('map_actors.*')).read_text()
    map_actors = json.loads(map_actor_text)
    # Take last actor data index as representative time step index
    scenario = Scenario(pn, map_actors, rng_seed=rng_seed, time_range=time_range)
    scenario.add_participants(participants)
    # save applied grid fee matrix
    results_path = cfg.config.results_path
    # if the market_results directory does not already exist, create it
    if not results_path.exists() and results_path:
        results_path.mkdir()
    results_path.joinpath('used_grid_fee_matrix.inf').write_text(json.dumps(pn.grid_fee_matrix))

    return scenario


def create_random(num_nodes, num_actors, weight_factor, nb_ts=100, horizon=24):
    # Create random nodes
    pn = power_network.create_random(num_nodes)

    # Update the shortest paths and the grid fee matrix
    pn.update_shortest_paths()
    pn.generate_grid_fee_matrix(weight_factor)
    mm_buy_prices = np.random.random(nb_ts+horizon)
    scenario = Scenario(pn, None, buy_prices=mm_buy_prices)
    actors = [actor.create_random("H" + str(i), nb_ts=nb_ts, horizon=horizon)
              for i in range(num_actors)]
    # Quick fix set Timestemps range
    scenario.environment.time_range = actors[0].data.index

    # Add actor nodes at random position (leaf node) in the network
    # One network node can contain several actors (using random.choices method)
    scenario.map_actors = pn.add_actors_random(actors)
    scenario.add_participants(actors)
    return scenario


def create_random2(num_nodes, num_actors, nb_ts=100, horizon=24):
    assert num_actors < num_nodes
    # num_actors has to be much smaller than num_nodes
    # Create random nodes
    pn = power_network.create_random(num_nodes)

    # Create random actors
    actors = [actor.create_random("H" + str(i), nb_ts=nb_ts, horizon=horizon)
              for i in range(num_actors)]

    # Add actor nodes at random position (leaf node) in the network
    # One selected network node (using random.sample method), directly represents a single actor
    # No nodes and edges are added to the network
    actor_nodes = random.sample(pn.leaf_nodes, num_actors)
    map_actors = {actor.id: node_id for actor, node_id in zip(actors, actor_nodes)}

    mm_buy_prices = np.random.random(nb_ts+horizon)
    scenario = Scenario(pn, map_actors, mm_buy_prices)
    scenario.add_participants(actors)
    return scenario


def create_scenario_from_csv(dirpath, num_nodes, num_actors, weight_factor, ts_hour=4, nb_ts=None,
                             horizon=24):
    """
    Load csv files from path and randomly select num_actors to be randomly

    :param dirpath: Path object
    :param num_nodes: number of nodes in the network
    :param num_actors: number of actors in the network
    :param weight_factor: weight factor used to derive grid fees
    :param ts_hour: number of time slot of equal length within one hour
    :param nb_ts: number of time slots to be generated
    :param horizon: number of time slots to look into future to make the prediction for actor
        strategy
    """
    # Create random nodes in the power network
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
        a = actor.create_from_csv("H_" + str(i), asset_dict={
            "load": {"csv": filename, "col_index": 1},
            "pv": {}
        }, start_date="2016-01-01", nb_ts=nb_ts, horizon=horizon, ts_hour=ts_hour)

        actors.append(a)

    # Add actor nodes at random position (leaf node) in the network
    # One network node can contain several actors (using random.choices method)
    map_actors = pn.add_actors_random(actors)

    # Update the shortest paths and the grid fee matrix
    pn.update_shortest_paths()
    pn.generate_grid_fee_matrix(weight_factor)
    scenario = Scenario(pn, map_actors, steps_per_hour=ts_hour)
    scenario.add_participants(actors)
    return
