import networkx as nx
import pytest

from simply.actor import Actor
from simply.market_fair import BestMarket
from simply.power_network import PowerNetwork
from simply.scenario import Scenario
from simply import config as cfg
import pandas as pd
import numpy as np


class TestMultipleActors:
    np.random.seed(42)
    NUM_STEPS = 96
    df = pd.DataFrame(np.random.rand(NUM_STEPS, 2), columns=["load", "pv"])
    cfg.Config("")
    SELL_MULT = 1/0.8
    # Sell prices of are higher than buy_prices. This way the MarketMaker makes a profit
    @pytest.fixture()
    def scenario(self):
        nw = nx.Graph()
        nw.add_edges_from([(0, 1, {"weight": 1}), (1, 2), (1, 3), (0, 4)])
        pn = PowerNetwork("", nw, weight_factor=1)
        test_prices = [0.082, 0.083, 0.087, 0.102, 0.112, 0.122, 0.107, 0.103, 0.1, 0.1, 0.09,
                       0.082,
                       0.083, 0.083, 0.094, 0.1, 0.11, 0.109, 0.106, 0.105, 0.1, 0.093, 0.084,
                       0.081,
                       0.078, 0.074, 0.074, 0.079, 0.081, 0.083, 0.079, 0.074, 0.07, 0.067, 0.065,
                       0.067, 0.073, 0.075, 0.085, 0.095, 0.107, 0.107, 0.107, 0.107, 0.1, 0.094,
                       0.087,
                       0.08]
        scenario = Scenario(pn, None, buy_prices=np.tile(test_prices, 10), steps_per_hour=4,
                            sell_prices=np.tile(test_prices, 10)*self.SELL_MULT)
        scenario.add_market(BestMarket(pn))
        return scenario

    num_actors=0

    def create_actor(self):
        self.num_actors +=1
        return Actor(self.num_actors, self.df)

    def get_banks_no_interaction(self, scenario, actors):
        """
        Run a scenario without any interaction between actors. This is done by only having one
        actor as market participant, and repeatedly running the scenario
        :return:
        """
        banks={}
        for actor in actors:
            scenario.add_participant(actor)
            self.run_simply(scenario)
            banks[actor.id]=actor.bank
            scenario.reset()
        return banks

    def run_simply(self, scenario):
        for _ in range(self.NUM_STEPS):
            # actors calculate strategy based market interaction with the market maker
            scenario.create_strategies()
            # orders are generated based on the flexibility towards the planned market interaction
            # and a pricing scheme. Orders are matched at the end
            scenario.market_step()
            # actors are prepared for the next time step by changing socs, banks and predictions
            scenario.next_time_step()

    def test_no_interaction(self, scenario):
        actors = [self.create_actor() for _ in range(2)]
        banks=self.get_banks_no_interaction(scenario, actors)
        scenario.add_participants(actors)
        self.run_simply(scenario)
        for actor in actors:
            assert actor.bank == banks[actor.id]

    def test_interaction(self, scenario):
        cfg.config.default_grid_fee=0.1
        actors = [self.create_actor() for _ in range(2)]
        act = actors [0]
        act.data.loc[:,"load"] = np.roll(np.array(act.data.loc[:,"load"]), 6)
        banks = self.get_banks_no_interaction(scenario, actors)
        scenario.add_participants(actors)
        act.create_prediction()
        self.run_simply(scenario)
        for actor in actors:
            assert actor.bank == banks[actor.id]

