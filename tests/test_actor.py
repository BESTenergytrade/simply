import pandas as pd
import numpy as np
import pytest


from simply.actor import Actor, create_random, Order
from simply.battery import Battery
from simply.market_fair import BestMarket, MARKET_MAKER_THRESHOLD
from simply.power_network import PowerNetwork
import simply.config as cfg
import networkx as nx
from pytest import approx
from simply.scenario import Scenario

ratings = dict()
NR_STEPS = 100
SELL_MULT = 0.9
BAT_CAPACITY = 3


def actor_print(actor, header=False, _header=dict()):
    if not header or actor in _header:
        pass
    else:
        header = ("Battery Energy, "
                  "Actor Schedule, "
                  "Actor Market Schedule, "
                  "Battery SOC, "
                  "Actor Bank, "
                  "Buying Price, "
                  "Matched Energy")
        print(header)
    _header[actor] = True

    print(f"{actor.t},"
          f"{round(actor.battery.energy(),4)}, "
          f"{round(actor.pred.schedule[0],4)}, "
          f"{round(actor.market_schedule[0],4)}, "
          f"{round(actor.battery.soc,4)}, "
          f"{round(actor.bank,4)},"
          f"{round(actor.pred.price[0],4)},"
          f"{round(actor.matched_energy_current_step,4)}")


def market_step(actor, market, step_time):
    order = actor.generate_order()
    # Get the order into the market
    market.accept_order(order, callback=actor.receive_market_results)
    # Generate market maker order as ask
    market.accept_order(
        Order(1, step_time, 'market_maker', None, MARKET_MAKER_THRESHOLD, actor.pred.price[0]))
    # Generate market maker order as bid
    market.accept_order(
        Order(-1, step_time, 'market_maker', None, MARKET_MAKER_THRESHOLD,
              actor.pred.selling_price[0]))
    market.clear()


class TestActor:
    df = pd.DataFrame(np.random.rand(24, 4), columns=["load", "pv", "price", "schedule"])
    cfg.Config("")
    nw = nx.Graph()
    nw.add_edges_from([(0, 1, {"weight": 1}), (1, 2), (1, 3), (0, 4)])
    pn = PowerNetwork("", nw, weight_factor=1)
    test_prices = [0.082, 0.083, 0.087, 0.102, 0.112, 0.122, 0.107, 0.103, 0.1, 0.1, 0.09, 0.082,
                   0.083, 0.083, 0.094, 0.1, 0.11, 0.109, 0.106, 0.105, 0.1, 0.093, 0.084, 0.081,
                   0.078, 0.074, 0.074, 0.079, 0.081, 0.083, 0.079, 0.074, 0.07, 0.067, 0.065,
                   0.067, 0.073, 0.075, 0.085, 0.095, 0.107, 0.107, 0.107, 0.107, 0.1, 0.094, 0.087,
                   0.08]

    test_schedule = [0.164, 0.077, 0.019, -0.038, -0.281, -0.054, -0.814, -1.292, -1.301, -1.303,
                     -1.27, -1.228, -1.301, -0.392, -0.564, -0.411, 1.046, 1.385, 1.448, 1.553,
                     0.143, 0.123, 0.172, 0.094, 0.084, 0.075, -0.017, -0.071, -0.147, -0.23,
                     -1.208, -1.072, -0.277, -0.274, -0.813, -0.131, -0.844, 0.013, 0.071, -0.027,
                     -0.005, 1.08, 1.065, 1.406, 1.403, 1.341, 1.096, 1.098]

    # Note: Positive load values lead to negative schedule values.
    # Positive PV values lead to positive schedule values
    example_df = pd.DataFrame(
        list(zip([abs(val) if val < 0 else 0 for val in test_schedule],
                 [val if val > 0 else 0 for val in test_schedule],
                 test_schedule, test_prices)),
        columns=['load', 'pv', 'schedule', 'price'])
    example_df = pd.concat([example_df] * 10).reset_index(drop=True)

    def test_init(self):
        # actor_id, dataframe, load_scale, power_scale, pm (?)
        a = Actor(0, self.df)
        assert a is not None

    def test_generate_orders(self):
        a = Actor(0, self.df)
        o = a.generate_order()
        assert len(a.orders) == 1
        assert o.actor_id == 0

    def test_recv_market_results(self):
        # time, sign, energy, price
        a = Actor(0, self.df)
        o = a.generate_order()
        assert o.time == 0
        a.receive_market_results(o.time, o.type, o.energy / 4, o.price)
        # must be tuple at key 0 (current time)
        assert len(a.traded[0]) == 2
        # must have one energy and one price
        assert len(a.traded[0][0]) == 1
        assert len(a.traded[0][1]) == 1
        # double matches should not overwrite each other
        a.receive_market_results(o.time, o.type, o.energy / 2, o.price)
        # must still be tuple
        assert len(a.traded[0]) == 2
        # tuple must now contain lists of len 2
        assert len(a.traded[0][0]) == 2
        assert len(a.traded[0][1]) == 2

    def test_create_random(self):
        a = create_random(0)
        assert a.id == 0

    def test_create_random_multidays_halfhour(self):
        data_cfg = {"nb_ts": 96, "ts_hour": 2}
        a = create_random(0, **data_cfg)
        # time series is longer than one day
        assert a.data.index[0].date() != a.data.index[-1].date()

    def test_no_strategy(self):
        # overwrite strategy 0 with zero buys/sells. Assert that the simulation throws an error
        pn = PowerNetwork("", nx.random_tree(1))

        scenario = Scenario(pn, [], None, steps_per_hour=4)
        battery = Battery(
            capacity=BAT_CAPACITY, max_c_rate=2, soc_initial=0.0, check_boundaries=True)
        actor = Actor(0, self.example_df, battery=battery, scenario=scenario)
        m = BestMarket(0, self.pn)

        # make sure error is thrown
        with pytest.raises(AssertionError):
            # iterate over time steps
            for t in range(NR_STEPS):
                m.t = t
                actor.get_market_schedule(strategy=0)

                # nullify market schedule
                actor.market_schedule *= 0
                market_step(actor, m, t)

                for a in scenario.actors:
                    # update all actors for the next market time slot
                    a.next_time_step()

    def test_rule_based_strategy_0(self):
        # the simplest strategy which buys or sells exactly the amount of the schedule at the time
        # the energy is needed. A battery is still needed since schedule values do not necessarily
        # line up with the traded_energy amount, e.g. schedule does not have 0.01 steps.
        pn = PowerNetwork("", nx.random_tree(1))

        scenario = Scenario(pn, [], None, steps_per_hour=4)
        battery = Battery(
            capacity=BAT_CAPACITY, max_c_rate=2, soc_initial=0.0, check_boundaries=True)
        actor = Actor(0, self.example_df, battery=battery, scenario=scenario)
        actor.data.selling_price *= SELL_MULT
        actor.create_prediction()

        m = BestMarket(0, self.pn)
        nr_of_matches = 0
        # iterate through time steps
        for t in range(NR_STEPS):
            m.t = t
            actor.get_market_schedule(strategy=0)
            market_step(actor, m, t)

            # tolerance due to energy_unit differences
            tol = 2*actor.energy_unit
            assert actor.battery.energy() < tol
            assert -actor.market_schedule[0] == approx(actor.pred.schedule[0], abs=tol)
            assert len(m.matches)-1 == nr_of_matches
            nr_of_matches = len(m.matches)

            for a in scenario.actors:
                # update all actors for the next market time slot
                a.next_time_step()

        ratings["strategy_0"] = actor.bank

    def test_rule_based_strategy_1(self):
        # strategy 1 buys energy at the lowest possible price before or exactly when energy is
        # predicted to be needed.
        # test that strategy 1 works without errors. Assert that price of energy is lower than
        # buying only in the current time slots
        pn = PowerNetwork("", nx.random_tree(1))

        scenario = Scenario(pn, [], None, steps_per_hour=4)
        battery = Battery(
            capacity=BAT_CAPACITY, max_c_rate=2, soc_initial=0.0, check_boundaries=True)
        actor = Actor(0, self.example_df, battery=battery, scenario=scenario)
        actor.data.selling_price *= SELL_MULT
        actor.create_prediction()
        m = BestMarket(0, self.pn)
        nr_of_matches = 0

        cost_no_strat = 0
        cost_with_strat = 0
        energy_no_strat = 0
        energy_with_strat = 0

        # iterate over time steps
        for t in range(NR_STEPS):
            m.t = t
            actor.get_market_schedule(strategy=1)
            market_step(actor, m, t)

            cost_no_strat += (actor.pred.schedule[0] < 0) * \
                -actor.pred.schedule[0] * actor.pred.price[0]
            energy_no_strat += (actor.pred.schedule[0] < 0) * \
                -actor.pred.schedule[0]

            # market schedule has opposite sign to schedule, i.e. positive sign schedule is pv
            # production which leads to negative sign market schedule
            cost_with_strat += (actor.market_schedule[0] > 0) * \
                actor.market_schedule[0] * actor.pred.price[0]
            energy_with_strat += (actor.market_schedule[0] > 0) * actor.market_schedule[0]

            assert len(m.matches)-1 == nr_of_matches
            nr_of_matches = len(m.matches)

            # battery makes sure soc bounds are not violated, so no assertion is needed here
            # make sure the schedule is planning only to buy energy. Might sell energy in the
            # current time steps to get rid of energy at SOC ~ 1
            if actor.pred.schedule[0] > 0:
                assert actor.pred.schedule[0] + actor.market_schedule[0] >= 0
            assert all(actor.market_schedule[1:] >= 0)
            actor.next_time_step()

        # make sure energy was bought for a lower price than just buying energy when it is needed.
        assert cost_with_strat < cost_no_strat
        # make sure the less energy is bought with strategy 1 than is bought without a
        # strategy since strategy 1 uses pv when possible
        # this should be the case in the current test scenario.
        assert energy_no_strat >= energy_with_strat
        minimal_price = actor.data.selling_price.min()

        cost_in_bat = minimal_price * actor.battery.energy()
        # battery could be full. If this energy would be sold for the minimal price strategy 1 has
        # to have a lower cost than strategy 0

        # average energy price of strategy 1 should be lower than no strategy.
        assert (cost_with_strat - cost_in_bat) / energy_with_strat < cost_no_strat / energy_no_strat
        ratings["strategy_1"] = actor.bank

    def test_rule_based_strategy_2(self):
        # strategy 2 extends strategy 1. It sells energy at the highest possible price before or
        # exactly when the battery would reach an soc of 1 or higher.
        # Assert that price of energy is lower than
        # buying only in the current time slots
        battery = Battery(capacity=BAT_CAPACITY, max_c_rate=2, soc_initial=0.0)
        actor = Actor(0, self.example_df, battery=battery, _steps_per_hour=4)
        actor.data.selling_price *= SELL_MULT
        actor.create_prediction()
        m = BestMarket(0, self.pn)
        nr_of_matches = 0
        # iterate over time steps
        for t in range(NR_STEPS):
            m.t = t
            actor.get_market_schedule(strategy=2)
            market_step(actor, m, t)
            assert len(m.matches)-1 == nr_of_matches
            nr_of_matches = len(m.matches)

            actor.next_time_step()
        ratings["strategy_2"] = actor.bank
        minimal_price = actor.data.selling_price.min()
        bank_in_bat = minimal_price * actor.battery.energy()
        # battery could be full. If this energy would be sold for the minimal price strategy 2 has
        # to have a higher bank than strategy 0
        assert ratings["strategy_2"] + bank_in_bat > ratings["strategy_0"]

    def test_rule_based_strategy_3(self):
        # strategy 3 extends strategy 2. It buys energy at low prices and sells it at high prices
        # if profit can be made and the soc of of the previous strategies allows for it.
        # Assert this strategy runs without errors.
        battery = Battery(capacity=BAT_CAPACITY, max_c_rate=2, soc_initial=0.0)
        actor = Actor(0, self.example_df, battery=battery, _steps_per_hour=4)
        actor.data.selling_price *= SELL_MULT
        actor.create_prediction()

        m = BestMarket(0, self.pn)
        nr_of_matches = 0
        # iterate over time steps
        for t in range(NR_STEPS):
            m.t = t
            actor.get_market_schedule(strategy=3)
            market_step(actor, m, t)
            assert len(m.matches)-1 == nr_of_matches
            nr_of_matches = len(m.matches)
            actor.next_time_step()
        ratings["strategy_3"] = actor.bank

    def test_strategy_ranking(self):
        # Assert that the strategy extensions improve the rating, e.g. bank account
        assert ratings["strategy_3"] >= ratings["strategy_2"] >= ratings["strategy_1"]

    def test_reduced_bank(self):
        # check that reducing the selling prices reduces profit
        battery = Battery(capacity=BAT_CAPACITY, max_c_rate=2, soc_initial=0.0)
        actor = Actor(0, self.example_df, battery=battery, _steps_per_hour=4)
        actor.data.selling_price *= SELL_MULT*0.5
        actor.create_prediction()
        m = BestMarket(0, self.pn)

        # iterate over time steps
        for t in range(NR_STEPS):
            m.t = t
            actor.get_market_schedule(strategy=3)
            market_step(actor, m, t)
            actor.next_time_step()
        assert ratings["strategy_3"] >= actor.bank

    def test_strategy_3_no_schedule(self):
        # without schedule and with no price difference, the profit an actor can make, is dependent
        # on the cumulated sum of positive price gradients and the battery capacity
        battery = Battery(capacity=BAT_CAPACITY, max_c_rate=2, soc_initial=0.0)
        actor = Actor(0, self.example_df, battery=battery, _steps_per_hour=4)
        actor.data.selling_price = actor.data.selling_price.copy()
        actor.data.schedule *= 0
        actor.data.pv *= 0
        actor.data.load *= 0
        actor.create_prediction()
        m = BestMarket(0, self.pn)
        # number of steps depends on data input. assertion only works if last step ends on price
        # maximum, since only then the actor makes use of stored energy by selling it
        NR_STEPS = 41
        # iterate over time steps
        for t in range(NR_STEPS):
            m.t = t
            actor.get_market_schedule(strategy=3)
            market_step(actor, m, t)
            actor.next_time_step()

        val = (self.example_df.price.diff()[:NR_STEPS]
               [self.example_df.price.diff()[:NR_STEPS] > 0].sum()*BAT_CAPACITY)
        assert val == approx(actor.bank)

    def test_pricing(self):
        battery = Battery(capacity=10, max_c_rate=2, soc_initial=0.5)
        actor = Actor(0, self.example_df, battery=battery, _steps_per_hour=4, pricing_strategy=None)
        buy_price = 10
        sell_price = 1
        actor.pred.price = buy_price
        actor.pred.selling_price = sell_price
        actor.pred.schedule[:] = 0
        actor.get_market_schedule(strategy=0)

        check_index = 5
        energy_amount = 1

        # If no pricing strategy is given, no order is generated, since the current time step
        # does not plan interaction
        actor.pricing_strategy = None
        actor.market_schedule[:] = 0
        actor.market_schedule[check_index] = energy_amount
        order = actor.generate_order()
        assert order is None

        # A pricing_strategy can ba a function with the inputs index, meaning time steps until some
        # market interaction in planned in the market_schedule, the price this interaction would use
        # (e.g. selling or buying price) and the energy amount, which is positive when energy is
        # bought
        # Note that the energy used is the adjusted and limited energy which is the energy
        # in the order. this amount is limited by battery capacity
        actor.pricing_strategy = lambda index, final_price, energy: final_price/index + energy
        order = actor.generate_order()
        assert order.price == buy_price/check_index + energy_amount
        actor.market_schedule[:] = 0
        energy_amount = - energy_amount
        actor.market_schedule[check_index] = energy_amount
        order = actor.generate_order()
        assert order.price == sell_price/check_index + energy_amount

        # linear pricing function increases or decreases the order price by the given factor.
        # If the market schedule dictates buying power in 4 time steps for 1$ the current price
        # for order generation is set to 0.6$. If energy is sold the price would be increased
        # instead.
        actor.pricing_strategy = dict(name="linear", param=[0.1])
        check_index = 4
        actor.market_schedule[:] = 0
        actor.market_schedule[check_index] = 1
        order = actor.generate_order()
        assert order.price == buy_price - check_index*0.1*buy_price

        actor.market_schedule[check_index] = -1
        order = actor.generate_order()
        assert order.price == sell_price + check_index*0.1*sell_price

        check_index = 0
        actor.market_schedule[:] = 0
        actor.market_schedule[check_index] = 1
        order = actor.generate_order()
        assert order.price == buy_price

        actor.market_schedule[check_index] = -1
        order = actor.generate_order()
        assert order.price == sell_price




        # negative prices are possible when buying
        actor.market_schedule[:] = 0
        actor.market_schedule[20] = 1
        order = actor.generate_order()
        assert order.price < 0

        # harmonic pricing changes the price according to the harmonic series meaning
        # 1, 1/2, 1/3, 1/4, 1/5 ... and so on. The mandatory parameter is the half life index
        # which dictates at which index 1/2 is reached the function follows the formula
        # final_price * ((index / half_life_index) + 1) ** (- sign(energy))
        #
        actor.pricing_strategy = dict(name="harmonic", param=[3])
        check_index = 0
        actor.market_schedule[:] = 0
        actor.market_schedule[check_index] = 1
        order = actor.generate_order()
        assert order.price == buy_price
        actor.market_schedule[check_index] = -1
        order = actor.generate_order()
        assert order.price == sell_price

        check_index = 3
        actor.market_schedule[:] = 0
        actor.market_schedule[check_index] = 1
        order = actor.generate_order()
        assert order.price == buy_price / 2



        actor.market_schedule[check_index] = -1
        order = actor.generate_order()
        assert order.price == sell_price * 2

        check_index = 6
        actor.market_schedule[:] = 0
        actor.market_schedule[check_index] = 1
        order = actor.generate_order()
        assert order.price == pytest.approx(buy_price / 3)

        actor.market_schedule[check_index] = -1
        order = actor.generate_order()
        assert order.price == sell_price * 3

        actor.pricing_strategy = dict(name="harmonic", param=[1, 0.6])
        check_index = 15
        actor.market_schedule[:] = 0
        actor.market_schedule[check_index] = 1
        order = actor.generate_order()
        assert order.price == 6.25

        actor.market_schedule[:] = 0
        actor.market_schedule[check_index] = -1
        order = actor.generate_order()
        assert order.price == 1.6

        actor.market_schedule[:] = 0
        actor.market_schedule[0] = 1
        order = actor.generate_order()
        assert order.price == buy_price
        # ToDo test geometric series
        order = actor.generate_order()
        actor.pricing_strategy = dict(name="harmonic", param=[1, 0.6])
        order = actor.generate_order()
        actor.pricing_strategy = dict(name="harmonic", param=[0.9,0.5])
        order = actor.generate_order()


TestActor().test_pricing()