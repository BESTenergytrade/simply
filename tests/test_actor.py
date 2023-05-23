import pandas as pd
import numpy as np
import pytest

from simply.actor import Actor, create_random, Order
from simply.battery import Battery
from simply.market_fair import BestMarket
from simply.power_network import PowerNetwork
import simply.config as cfg
import networkx as nx
from pytest import approx
from simply.scenario import Scenario

ratings = dict()
NR_STEPS = 100
SELL_MULT = 0.9
BAT_CAPACITY = 3


# def market_step(actor, market, step_time):
#     order, = actor.generate_orders()
#     # Get the order into the market
#     market.accept_order(order, callback=actor.receive_market_results)
#     # Generate market maker order as ask
#     market.accept_order(
#         Order(1, step_time, 'market_maker', None, MARKET_MAKER_THRESHOLD, actor.pred.price[0]))
#     # Generate market maker order as bid
#     market.accept_order(
#         Order(-1, step_time, 'market_maker', None, MARKET_MAKER_THRESHOLD,
#               actor.pred.selling_price[0]))
#     market.clear()


class StubClass:
    def __init__(self):
        pass


class TestActor:
    stub_scenario = StubClass()
    stub_scenario.time_step = 0
    stub_scenario.actors = []

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

    scenario = Scenario(pn, [], None, buy_prices=np.tile(test_prices, 10), steps_per_hour=4)
    env = scenario.environment

    market = BestMarket(scenario, pn)

    test_schedule = [0.164, 0.077, 0.019, -0.038, -0.281, -0.054, -0.814, -1.292, -1.301, -1.303,
                     -1.27, -1.228, -1.301, -0.392, -0.564, -0.411, 1.046, 1.385, 1.448, 1.553,
                     0.143, 0.123, 0.172, 0.094, 0.084, 0.075, -0.017, -0.071, -0.147, -0.23,
                     -1.208, -1.072, -0.277, -0.274, -0.813, -0.131, -0.844, 0.013, 0.071, -0.027,
                     -0.005, 1.08, 1.065, 1.406, 1.403, 1.341, 1.096, 1.098]

    # Note: Splitting schedule into load and generation:
    # Positive load values lead to negative schedule values.
    # Positive PV values lead to positive schedule values
    example_df = pd.DataFrame(
        list(zip([abs(val) if val < 0 else 0 for val in test_schedule],
                 [val if val > 0 else 0 for val in test_schedule],
                 test_schedule)),
        columns=['load', 'pv', 'schedule'])
    example_df = pd.concat([example_df] * 10).reset_index(drop=True)

    def test_init(self):
        # actor_id, dataframe, load_scale, power_scale, pm (?)
        a = Actor(0, self.df, self.env)
        assert a is not None

    def test_generate_orders(self):
        a = Actor(0, self.df, self.env)
        o = a.generate_orders()
        assert len(o) == 1

        assert len(a.orders) == 1
        assert o[0].actor_id == 0

    def test_recv_market_results(self):
        # time, sign, energy, price
        a = Actor(0, self.df, self.env)
        o, = a.generate_orders()
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
        a = create_random(0, self.env)
        assert a.id == 0

    def test_create_random_multidays_halfhour(self):
        data_cfg = {"nb_ts": 96, "ts_hour": 2}
        a = create_random(0, self.env, **data_cfg)
        # time series is longer than one day
        assert a.data.index[0].date() != a.data.index[-1].date()

    def test_default_battery(self):
        actor = Actor(0, self.example_df, self.env, battery=None)
        assert actor.battery is not None
        assert isinstance(actor.battery, Battery)
        assert actor.battery.capacity == cfg.config.energy_unit*2

        actor = Actor(0, self.example_df, environment=self.env)
        assert actor.battery is not None
        assert isinstance(actor.battery, Battery)
        assert actor.battery.capacity == cfg.config.energy_unit*2

        battery = Battery(
            capacity=BAT_CAPACITY, max_c_rate=2, soc_initial=0.0, check_boundaries=True)
        actor = Actor(0, self.example_df, battery=battery, environment=self.env)
        assert actor.battery is not None
        assert isinstance(actor.battery, Battery)
        assert actor.battery.capacity == BAT_CAPACITY

    def test_no_strategy(self):
        # overwrite strategy 0 with zero buys/sells. Assert that the simulation throws an error
        battery = Battery(
            capacity=BAT_CAPACITY, max_c_rate=2, soc_initial=0.0, check_boundaries=True)
        actor = Actor(0, self.example_df, battery=battery, environment=self.env)

        # Check if UserWarning correctly prints out the strategy name if the strategy is not found
        foo = "foo"
        with pytest.warns(UserWarning, match=foo):
            actor.get_market_schedule(strategy=foo)

        # having an actor strategy should not overwrite the explicit strategy call
        actor.strategy = 3
        with pytest.warns(UserWarning, match=foo):
            actor.get_market_schedule(strategy=foo)

        # assert that the strategy used when no strategy is given as argument, that the default
        # market_schedule is different
        strategy_3 = actor.get_market_schedule().copy()
        strategy_0_explicit = actor.get_market_schedule(strategy=0).copy()
        actor.strategy = 0
        strategy_0_implicit = actor.get_market_schedule().copy()

        # at least one value has to be different
        assert (strategy_0_explicit != strategy_3).any()

        # all values must be equal
        assert (strategy_0_implicit == strategy_0_explicit).all()

    def test_rule_based_strategy_0(self):
        # the simplest strategy which buys or sells exactly the amount of the schedule at the time
        # the energy is needed. A battery is still needed since schedule values do not necessarily
        # line up with the traded_energy amount, e.g. schedule does not have 0.01 steps.

        self.scenario.reset()
        battery = Battery(
            capacity=BAT_CAPACITY, max_c_rate=2, soc_initial=0.0, check_boundaries=True)
        # self.scenario.reset()
        actor = Actor(0, self.example_df, battery=battery, environment=self.env)

        self.env.market_maker.all_buy_prices *= SELL_MULT
        self.env.market_maker.create_prediction()

        # self.scenario.market = m
        nr_of_matches = 0
        # tolerance due to energy_unit differences
        tol = 2 * cfg.config.energy_unit

        # iterate through time steps
        for _ in range(NR_STEPS):
            actor.get_market_schedule(strategy=0)

            # in principle the market_schedule has the same values as the schedule with opposite
            # signs. An energy need with negative sign in the schedule is met with buying energy in
            # the market_schedule which has a positive sign.
            assert -actor.market_schedule[0] == approx(actor.pred.schedule[0], abs=tol)
            self.scenario.market_step()

            # after the market step including matching and dis/charging the battery. the market
            # schedule is adjusted, meaning the market_schedule is reduced by the matched energy
            # Therefore the 1st element should be close to zero
            assert -actor.market_schedule[0] == approx(0, abs=tol)

            # strategy 0 is fulfilling the schedule just in time. therefore the battery should not
            # be in use, only minor fluctuations due to differences in between the energy unit
            # and the schedule are allowed.
            assert actor.battery.energy() < tol

            # make sure every iteration an order is placed and also matched. Pricing must guarantee
            # order fulfillment
            market = self.scenario.market
            assert len(market.matches)-1 == nr_of_matches, 'Generated order was not matched'
            nr_of_matches = len(market.matches)
            self.scenario.next_time_step()

        ratings["strategy_0"] = actor.bank

    def test_rule_based_strategy_1(self):
        # strategy 1 buys energy at the lowest possible price before or exactly when energy is
        # predicted to be needed.
        # test that strategy 1 works without errors. Assert that price of energy is lower than
        # buying only in the current time slots

        # reset scenario
        self.scenario.reset()
        battery = Battery(
            capacity=BAT_CAPACITY, max_c_rate=2, soc_initial=0.0, check_boundaries=True)
        actor = Actor(0, self.example_df, battery=battery, environment=self.env)

        self.env.market_maker.all_buy_prices *= SELL_MULT
        self.env.market_maker.create_prediction()

        nr_of_matches = 0
        cost_no_strat = 0
        cost_with_strat = 0
        energy_no_strat = 0
        energy_with_strat = 0

        # iterate over time steps
        for _ in range(NR_STEPS):
            actor.get_market_schedule(strategy=1)
            self.scenario.market_step()

            cost_no_strat += (actor.pred.schedule[0] < 0) * \
                -actor.pred.schedule[0] * self.env.market_maker.current_sell_price
            energy_no_strat += (actor.pred.schedule[0] < 0) * \
                -actor.pred.schedule[0]

            # market schedule has opposite sign to schedule, i.e. positive sign schedule is pv
            # production which leads to negative sign market schedule
            cost_with_strat += (actor.market_schedule[0] > 0) * \
                actor.market_schedule[0] * self.env.market_maker.current_sell_price
            energy_with_strat += (actor.market_schedule[0] > 0) * actor.market_schedule[0]

            assert len(self.scenario.market.matches)-1 == nr_of_matches
            nr_of_matches = len(self.scenario.market.matches)

            # battery makes sure soc bounds are not violated, so no assertion is needed here
            # make sure the schedule is planning only to buy energy. Might sell energy in the
            # current time steps to get rid of energy at SOC ~ 1
            if actor.pred.schedule[0] > 0:
                assert actor.pred.schedule[0] + actor.market_schedule[0] >= 0
            assert all(actor.market_schedule[1:] >= 0)
            self.scenario.next_time_step()

        # make sure energy was bought for a lower price than just buying energy when it is needed.
        assert cost_with_strat < cost_no_strat
        # make sure the less energy is bought with strategy 1 than is bought without a
        # strategy since strategy 1 uses pv when possible
        # this should be the case in the current test scenario.
        assert energy_no_strat >= energy_with_strat
        minimal_price = self.env.market_maker.all_buy_prices.min()

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
        # reset scenario

        self.scenario.reset()
        battery = Battery(
            capacity=BAT_CAPACITY, max_c_rate=2, soc_initial=0.0, check_boundaries=True)
        actor = Actor(0, self.example_df, battery=battery, environment=self.env)

        self.env.market_maker.all_buy_prices *= SELL_MULT
        self.env.market_maker.create_prediction()

        nr_of_matches = 0
        # iterate over time steps
        market = self.scenario.market
        for _ in range(NR_STEPS):
            actor.get_market_schedule(strategy=2)
            self.scenario.market_step()
            assert len(market.matches)-1 == nr_of_matches
            nr_of_matches = len(market.matches)
            self.scenario.next_time_step()

        ratings["strategy_2"] = actor.bank
        minimal_price = self.env.market_maker.all_buy_prices.min()
        bank_in_bat = minimal_price * actor.battery.energy()
        # battery could be full. If this energy would be sold for the minimal price strategy 2 has
        # to have a higher bank than strategy 0
        assert ratings["strategy_2"] + bank_in_bat > ratings["strategy_0"]

    def test_rule_based_strategy_3(self):
        # strategy 3 extends strategy 2. It buys energy at low prices and sells it at high prices
        # if profit can be made and the soc of of the previous strategies allows for it.
        # Assert this strategy runs without errors.
        battery = Battery(capacity=BAT_CAPACITY, max_c_rate=2, soc_initial=0.0)
        self.scenario.reset()
        actor = Actor(0, self.example_df, environment=self.env, battery=battery)

        self.env.market_maker.all_buy_prices *= SELL_MULT
        self.env.market_maker.create_prediction()

        market = self.scenario.market
        nr_of_matches = 0
        # iterate over time steps
        for _ in range(NR_STEPS):
            actor.get_market_schedule(strategy=3)
            self.scenario.market_step()
            assert len(market.matches)-1 == nr_of_matches
            nr_of_matches = len(market.matches)
            self.scenario.next_time_step()
        ratings["strategy_3"] = actor.bank

    def test_strategy_ranking(self):
        # Assert that the strategy extensions improve the rating, e.g. bank account
        assert ratings["strategy_3"] >= ratings["strategy_2"] >= ratings["strategy_1"]

    def test_reduced_bank(self):
        # check that reducing the selling prices (mm_buy_prices) reduces profit
        self.scenario.reset()
        battery = Battery(capacity=BAT_CAPACITY, max_c_rate=2, soc_initial=0.0)
        actor = Actor(0, self.example_df, environment=self.env, battery=battery)
        self.env.market_maker.all_buy_prices *= SELL_MULT / 2
        self.env.market_maker.create_prediction()

        # iterate over time steps
        for _ in range(NR_STEPS):
            actor.get_market_schedule(strategy=3)
            self.scenario.market_step()
            self.scenario.next_time_step()
        assert ratings["strategy_3"] >= actor.bank

    def test_strategy_3_no_schedule(self):
        # without schedule and with no price difference, the profit an actor can make is dependent
        # on the cumulated sum of positive price gradients and the battery capacity
        self.scenario.reset()
        battery = Battery(capacity=BAT_CAPACITY, max_c_rate=2, soc_initial=0.0)
        actor = Actor(0, self.example_df, environment=self.env, battery=battery)
        market_maker = self.env.market_maker
        market_maker.all_sell_prices = market_maker.all_buy_prices.copy()
        market_maker.create_prediction()

        actor.data.schedule *= 0
        actor.data.pv *= 0
        actor.data.load *= 0
        actor.create_prediction()
        # number of steps depends on data input. assertion only works if last step ends on price
        # maximum, since only then the actor makes use of stored energy by selling it
        NR_STEPS = 41
        # iterate over time steps
        for _ in range(NR_STEPS):
            actor.get_market_schedule(strategy=3)
            self.scenario.market_step()
            self.scenario.next_time_step()
        sell_prices = self.env.market_maker.all_sell_prices
        val = (np.diff(sell_prices)[:NR_STEPS][np.diff(sell_prices)[:NR_STEPS] > 0].sum()
               * BAT_CAPACITY)
        assert val == approx(actor.bank)

    def test_limiting_energy(self):
        # test limit of energy amount
        battery = Battery(capacity=10, max_c_rate=2, soc_initial=0.99)
        # with this soc and capacity it means max 0.1 energy can be stored
        actor = Actor(0, self.example_df, environment=self.env, battery=battery)
        actor.pricing_strategy = dict(name="linear", param=[0.1])
        actor.pred.schedule[:] = 0
        actor.get_market_schedule(strategy=0)
        actor.market_schedule[:] = 0
        # the market plan is to buy 10 energy in 5 time steps in the future
        actor.market_schedule[5] = 10
        orders = actor.generate_orders()
        # the order energy is limited to 1% capacity of the battery which is initialized at 99%
        assert orders[0].energy == pytest.approx(0.1)

        # vice versa if the plan was to sell energy only 9.9 energy could be sold at this point
        actor.pred.schedule[:] = 0
        actor.get_market_schedule(strategy=0)
        actor.market_schedule[:] = 0
        actor.market_schedule[5] = -10
        orders = actor.generate_orders()
        assert orders[0].energy == pytest.approx(9.9)

    def test_pricing(self):
        # Test different pricing algorithms as well as self defined pricing functionality.
        # Test both cases of buying and selling energy, since pricing schemes lower the price
        # in comparison to the guaranteed buying price and raise the price in comparison to the
        # guaranteed selling price
        self.scenario.reset()
        bat = Battery(capacity=10, max_c_rate=2, soc_initial=0.5)
        actor = Actor(0, self.example_df, environment=self.env, battery=bat, pricing_strategy=None)
        buy_price = 10
        sell_price = 1
        self.env.market_maker.all_sell_prices[:] = buy_price
        self.env.market_maker.all_buy_prices[:] = sell_price
        self.env.market_maker.create_prediction()

        actor.pred.schedule[:] = 0
        actor.get_market_schedule(strategy=0)

        check_index = 5
        energy_amount = 1

        # If no pricing strategy is given, no order is generated, since the current time step
        # does not plan interaction. The planned order at index 5 is ignored.
        actor.pricing_strategy = None
        actor.market_schedule[:] = 0
        actor.market_schedule[check_index] = energy_amount
        orders = actor.generate_orders()
        assert len(orders) == 0

        # custom pricing strategy
        # A pricing_strategy can ba a function with the inputs index, meaning time steps until some
        # market interaction is planned in the market_schedule, the price this interaction would use
        # (e.g. selling or buying price) and the energy amount, which is positive when energy is
        # bought
        # Note that the energy used is the adjusted and limited energy which is the energy
        # in the order. this amount is limited by battery capacity
        actor.pricing_strategy = lambda index, final_price, energy: final_price/index + energy
        orders = actor.generate_orders()
        assert orders[0].price == buy_price/check_index + energy_amount
        actor.market_schedule[:] = 0
        energy_amount = - energy_amount
        actor.market_schedule[check_index] = energy_amount
        orders = actor.generate_orders()
        assert orders[0].price == sell_price/check_index + energy_amount

        # linear pricing function increases or decreases the order price by the given factor.
        # If the market schedule dictates buying power in 4 time steps for 1$, the current price
        # for order generation is set to 0.6$ since we used 0.1 as gradient.
        # If energy is sold the price would be increased instead.
        actor.pricing_strategy = dict(name="linear", param=[0.1])
        check_index = 4
        actor.market_schedule[:] = 0
        actor.market_schedule[check_index] = 1
        orders = actor.generate_orders()
        assert orders[0].price == buy_price - check_index*0.1*buy_price

        actor.market_schedule[check_index] = -1
        orders = actor.generate_orders()
        assert orders[0].price == sell_price + check_index*0.1*sell_price

        # make sure final price is given when the next time step with energy needs is the current
        # time step
        check_index = 0
        actor.market_schedule[:] = 0
        actor.market_schedule[check_index] = 1
        orders = actor.generate_orders()
        assert orders[0].price == buy_price

        actor.market_schedule[check_index] = -1
        orders = actor.generate_orders()
        assert orders[0].price == sell_price

        # negative prices are possible when buying
        actor.market_schedule[:] = 0
        actor.market_schedule[20] = 1
        orders = actor.generate_orders()
        assert orders[0].price < 0

        # test harmonic pricing
        # harmonic pricing changes the price according to the harmonic series meaning
        # 1, 1/2, 1/3, 1/4, 1/5 ... and so on. The mandatory parameter is the half life index
        # which dictates at which index 1/2 is reached the function follows the formula
        # final_price * ((index / half_life_index) + 1) ** (- sign(energy))
        #
        actor.pricing_strategy = dict(name="harmonic", param=[3])
        # check that the final price is given for index 0
        check_index = 0
        actor.market_schedule[:] = 0
        actor.market_schedule[check_index] = 1
        orders = actor.generate_orders()
        assert orders[0].price == buy_price
        actor.market_schedule[check_index] = -1
        orders = actor.generate_orders()
        assert orders[0].price == sell_price

        # check that for index = half_life_index, the final price is halved
        check_index = 3
        actor.market_schedule[:] = 0
        actor.market_schedule[check_index] = 1
        orders = actor.generate_orders()
        assert orders[0].price == buy_price / 2

        # check that for index = half_life_index, the final price is doubled for selling
        actor.market_schedule[check_index] = -1
        orders = actor.generate_orders()
        assert orders[0].price == sell_price * 2

        # double the half_life_index means a third of the final_price
        check_index = 6
        actor.market_schedule[:] = 0
        actor.market_schedule[check_index] = 1
        orders = actor.generate_orders()
        assert orders[0].price == pytest.approx(buy_price / 3)

        # and vice versa for selling
        actor.market_schedule[check_index] = -1
        orders = actor.generate_orders()
        assert orders[0].price == sell_price * 3

        # harmonic pricing allows for a second parameter called symmetric_bound_factor. It leads to
        # the pricing converging towards factor*final_price, in this case 60% of the final price.
        # the upper bound symmetric so that it is bound to 1/0.6 --> 166% of the final price
        actor.pricing_strategy = dict(name="harmonic", param=[1, 0.6])
        check_index = 15
        actor.market_schedule[:] = 0
        actor.market_schedule[check_index] = 1
        orders = actor.generate_orders()
        assert orders[0].price == 6.25

        actor.market_schedule[:] = 0
        actor.market_schedule[check_index] = -1
        orders = actor.generate_orders()
        assert orders[0].price == 1.6
        # 1.6 / 1 =-> reciprocal value 0.625. Therefore 10 * 0.625 ==6.25

        # check that prices are still properly generated for current time step energy
        actor.market_schedule[:] = 0
        actor.market_schedule[0] = 1
        orders = actor.generate_orders()
        assert orders[0].price == buy_price

        actor.market_schedule[:] = 0
        actor.market_schedule[0] = -1
        orders = actor.generate_orders()
        assert orders[0].price == sell_price

        # test geometric series
        # this series has a constant factor in between the elements of the series, so that the
        # n-th element, with the geometric factor x and the start value n0 is defined as
        # n-th element = n0 * x^(n)
        actor.pricing_strategy = dict(name="geometric", param=[0.9])
        check_index = 3
        actor.market_schedule[:] = 0
        actor.market_schedule[check_index] = 1
        orders = actor.generate_orders()
        assert orders[0].price == pytest.approx(buy_price*0.9**3)

        actor.market_schedule[:] = 0
        actor.market_schedule[check_index] = -1
        orders = actor.generate_orders()
        assert orders[0].price == pytest.approx(sell_price * (1/0.9) ** 3)

        # a second parameter can be used to cap the price to a factor of the final price
        # the pricing is capped/clipped at a price of factor*final_price, in this case 70% of the
        # final price- the upper bound is symmetric so that it is bound to 1/0.7 --> 143% of the
        # final price
        actor.pricing_strategy = dict(name="geometric", param=[0.9, 0.7])
        check_index = 6
        actor.market_schedule[:] = 0
        actor.market_schedule[check_index] = 1
        orders = actor.generate_orders()
        assert orders[0].price == 0.7*buy_price

        actor.market_schedule[:] = 0
        actor.market_schedule[check_index] = -1
        orders = actor.generate_orders()
        assert orders[0].price == pytest.approx(1/0.7 * sell_price)

    def test_market_schedule_adjustment(self):
        # test if the market schedule is properly adjusted when orders are matched with the pricing
        # strategy.
        self.scenario.reset()
        market = self.scenario.market
        battery = Battery(capacity=10, max_c_rate=2, soc_initial=0.0)
        # with this soc and capacity it means max 0.1 energy can be stored
        actor = Actor(0, self.example_df, environment=self.env, battery=battery, cluster=0)
        actor.pricing_strategy = dict(name="linear", param=[0.1])
        actor.pred.schedule[:] = 0
        actor.pred.schedule[[5, 10, 15]] = 5
        actor.pred.schedule[[2, 12, 13]] = -5
        actor.get_market_schedule(strategy=0)
        assert actor.market_schedule[2] == 5
        orders = actor.generate_orders()
        for order in orders:
            market.accept_order(order, callback=actor.receive_market_results)

        # Generate  order as ask
        market.accept_order(Order(1, 0, 'other_actor', 0, 4, 0.0001))
        market.clear()

        # the order energy is limited by the order of "other_actor"
        assert actor.market_schedule[2] == pytest.approx(1)

        actor.market_schedule[1] = -5
        # order will only be generated if the soc actually has energy to sell
        actor.battery.charge(7)
        orders = actor.generate_orders()
        for order in orders:
            market.accept_order(order, callback=actor.receive_market_results)

        # Generate order as ask
        market.accept_order(Order(-1, 0, 'other_actor', 0, 4, 9999))
        market.clear()
        assert actor.market_schedule[1] == pytest.approx(-1)

    def test_predict_soc(self):
        # without schedule and with no price difference, the profit an actor can make, is dependent
        # on the cumulated sum of positive price gradients and the battery capacity
        self.scenario.reset()
        battery = Battery(capacity=10, soc_initial=1)
        actor = Actor(0, self.example_df, environment=self.env, battery=battery)
        actor.data.load[:] = actor.data.load * 0 + 1
        actor.data.pv = actor.data.pv * 0
        actor.data.schedule = actor.data.pv - actor.data.load
        actor.create_prediction()
        actor.get_market_schedule(strategy=0)
        actor.market_schedule[:] = 0
        socs = actor.predict_socs()

        # number of steps depends on data input. assertion only works if last step ends on price
        # maximum, since only then the actor makes use of stored energy by selling it
        NR_STEPS = 10
        # iterate over time steps

        # check if prediction of soc lines up with actual soc when only the schedule has values
        for t in range(NR_STEPS):
            self.scenario.market_step()
            self.scenario.next_time_step()
            assert socs[t] == pytest.approx(actor.battery.soc)

        # check if prediction of soc lines up with actual soc when the schedule and market schedule
        # has values
        self.scenario.reset()
        battery = Battery(capacity=20, soc_initial=1)
        actor = Actor(0, self.example_df, self.env, battery=battery)
        actor.data.load[:] = actor.data.load * 0 + 1
        actor.data.pv = actor.data.pv * 0
        actor.data.schedule = actor.data.pv - actor.data.load
        actor.create_prediction()
        actor.get_market_schedule(strategy=0)
        actor.market_schedule[:] = -1
        socs = actor.predict_socs()
        # iterate over time steps
        for t in range(NR_STEPS):
            self.scenario.market_step()
            # reset market_schedule
            actor.market_schedule[:] = -1
            self.scenario.next_time_step()
            assert socs[t] == pytest.approx(actor.battery.soc, abs=1e-12)
            print(socs[t], actor.battery.soc)

        # check production and clipping
        self.scenario.reset()
        battery = Battery(capacity=10, soc_initial=0)
        actor = Actor(0, self.example_df, self.env, battery=battery)
        actor.data.load[:] = actor.data.load * 0
        actor.data.pv = actor.data.pv * 0 + 1
        actor.data.schedule = actor.data.pv - actor.data.load
        actor.create_prediction()
        actor.get_market_schedule(strategy=0)
        actor.market_schedule[:] = 0
        socs = actor.predict_socs()
        assert max(socs) > 1
        socs_clipped = actor.predict_socs(clip=True, clip_value=1)
        assert max(socs_clipped) <= 1

        # iterate over time steps
        for t in range(NR_STEPS):
            self.scenario.market_step()
            self.scenario.next_time_step()
            assert socs[t] == pytest.approx(actor.battery.soc, abs=1e-12)

        # check if planning horizon is working as intended
        planning_horizon = 12
        socs = actor.predict_socs(planning_horizon=planning_horizon)
        assert len(socs) == planning_horizon+1

    def test_clip_soc(self):
        self.scenario.reset()
        battery = Battery(capacity=10, soc_initial=0)
        actor = Actor(0, self.example_df, environment=self.env, battery=battery)
        # constant load leads to a decrease after clipping
        actor.data.load[:] = actor.data.load * 0 + 1
        # punctual generation exceeds battery capacity, otherwise 0
        actor.data.pv = actor.data.pv * 0
        actor.data.pv[[0, 1, 5, 7]] = 11
        actor.data.schedule = actor.data.pv - actor.data.load
        actor.create_prediction()
        actor.get_market_schedule(strategy=0)
        actor.market_schedule[:] = 0

        socs = actor.predict_socs(clip=True, clip_value=1)
        # Values of soc are bound to 1. Overcharge is not possible. Therefore soc drops below 1
        # right after production events at 0,1,5,7
        assert socs[0] == socs[1] == socs[5] == socs[7] == 1
        assert socs[2] == socs[6] < 1

        socs = actor.predict_socs(clip=False)
        # Values of soc are NOT bound to 1. Overcharge is possible. Therefore soc keeps on rising
        assert 1 == socs[0] < socs[1] < socs[5] < socs[7]
        assert 1 < socs[2] < socs[6]
