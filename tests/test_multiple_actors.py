from time import time
import pytest
import pandas as pd
import numpy as np

from simply.actor import Actor
from simply.market_2pac import TwoSidedPayAsClear
from simply.scenario import Scenario
from simply import config as cfg


class TestMultipleActors:
    np.random.seed(42)
    num_actors = 0
    NUM_STEPS = 24
    df = pd.DataFrame(np.random.rand(NUM_STEPS+24, 2), columns=["load", "pv"])
    cfg.Config("", "")
    df = df - df % cfg.config.energy_unit
    SELL_MULT = 1/0.8
    # Sell prices of are higher than buy_prices. This way the MarketMaker makes a profit

    @pytest.fixture()
    def scenario(self):
        test_prices = [0.082, 0.083, 0.087, 0.102, 0.112, 0.122, 0.107, 0.103, 0.1, 0.1, 0.09,
                       0.082,
                       0.083, 0.083, 0.094, 0.1, 0.11, 0.109, 0.106, 0.105, 0.1, 0.093, 0.084,
                       0.081,
                       0.078, 0.074, 0.074, 0.079, 0.081, 0.083, 0.079, 0.074, 0.07, 0.067, 0.065,
                       0.067, 0.073, 0.075, 0.085, 0.095, 0.107, 0.107, 0.107, 0.107, 0.1, 0.094,
                       0.087,
                       0.08]
        scenario = Scenario(None, None, buy_prices=np.tile(test_prices, 10), steps_per_hour=4,
                            sell_prices=np.tile(test_prices, 10)*self.SELL_MULT)
        scenario.add_market(TwoSidedPayAsClear(grid_fee_matrix=[[0, 1], [1, 0]]))
        return scenario

    def test_interaction(self, scenario):

        cfg.config.default_grid_fee = 0.1
        actor_strat = 2
        pricing_strategy = {"name": "linear", "param": [0.0]}
        capacity = 4
        num_actor = 2

        actors = [self.create_actor(cluster=0, capacity=capacity, load_factor=3)
                  for _ in range(num_actor)]
        actors[0].data.loc[:, "load"] = np.roll(np.array(actors[0].data.loc[:, "load"]), 6)

        actors.extend([self.create_actor(cluster=0, capacity=capacity, pv_factor=3)
                       for _ in range(num_actor)])
        actors[-1].data.loc[:, "load"] = np.roll(np.array(actors[0].data.loc[:, "pv"]), 3)
        for actor in actors:
            actor.strategy = actor_strat
            actor.pricing_strategy = pricing_strategy

        self.compare_interaction_w_strats(actor_strat, pricing_strategy, scenario, actors,
                                          capacity=capacity)

    def create_actor(self, cluster=None, capacity=None, soc=0, pv_factor=1, load_factor=1):
        self.num_actors += 1
        df = self.df.copy()
        df["load"] *= load_factor
        df["pv"] *= pv_factor
        actor = Actor(self.num_actors, df, cluster=cluster)
        if capacity is not None:
            actor.battery.capacity = capacity
            actor.battery.soc = soc
            actor.battery.soc_initial = soc
        return actor

    def get_banks_no_interaction(self, scenario, actors):
        """
        Run a scenario without any interaction between actors. This is done by only having one
        actor as market participant, and repeatedly running the scenario
        :return:
        """
        banks = {}
        scenario.reset()
        for actor in actors:
            scenario.add_participant(actor)
            run_simply(scenario, self.NUM_STEPS)
            banks[actor.id] = actor.bank
            scenario.reset()
        return banks

    def compare_interaction_w_strats(self, actor_strat, pricing_strategy, scenario, actors,
                                     capacity=None):
        scenario.reset()

        banks_only_mm = self.get_banks_no_interaction(scenario, actors)
        print("MM interaction banks: ", banks_only_mm)
        scenario.reset()
        scenario.add_participants(actors)
        run_simply(scenario, self.NUM_STEPS)
        banks = [actor.bank for actor in actors]
        print("Local interaction banks: ", banks)

        # With interaction the actors should get better or the same prices as if they had inter-
        # acted with only the MM
        # assert all([actor.bank - banks_only_mm[actor.id] >=0 for actor in actors])
        print("Profit from interaction with other actors ", [actor.bank - banks_only_mm[actor.id]
                                                             for actor in actors])
        print("MarketMaker sold ", sum(scenario.environment.market_maker.energy_sold))
        print("MarketMaker bought ", sum(scenario.environment.market_maker.energy_bought))


def run_simply(scenario, steps):
    for _ in range(steps):
        # actors calculate strategy based market interaction with the market maker
        scenario.create_strategies()
        # orders are generated based on the flexibility towards the planned market interaction
        # and a pricing scheme. Orders are matched at the end
        scenario.market_step()
        # actors are prepared for the next time step by changing socs, banks and predictions
        scenario.next_time_step()
    return {actor.id: actor.bank for actor in scenario.market_participants
            if isinstance(actor, Actor)}


def time_it(function, timers={}):
    """Decorator function to time the duration and number of function calls.

    :param function: function do be decorated
    :type function: function
    :param timers: storage for cumulated time and call number
    :type timers: dict
    :return: decorated function or timer if given function is None
    :rtype function or dict

    """
    if function:
        def decorated_function(*this_args, **kwargs):
            key = function.__name__
            start_time = time()
            return_value = function(*this_args, **kwargs)
            delta_time = time() - start_time
            try:
                timers[key]["time"] += delta_time
                timers[key]["calls"] += 1
            except KeyError:
                timers[key] = dict(time=0, calls=1)
                timers[key]["time"] += delta_time
            return return_value

        return decorated_function

    sorted_timer = dict(sorted(timers.items(), key=lambda x: x[1]["time"] / x[1]["calls"]))
    return sorted_timer
