import numpy as np
import pytest

from simply.market import ASK, BID, MARKET_MAKER_THRESHOLD, Market
from simply.market_maker import MarketMaker
import simply.config as cfg
from simply.scenario import Scenario
from simply.actor import create_random


class TestMarketMaker:
    cfg.Config("", "")
    buy_prices = np.arange(1, 100, 1)
    scenario = Scenario(None, None, None)
    env = scenario.environment

    def test_init(self):
        MarketMaker(buy_prices=self.buy_prices, environment=self.env)

    def test_price_comparison(self):
        MarketMaker(buy_prices=self.buy_prices, environment=self.env)
        # Assertion error should be thrown since mm would buy for higher prices than he would sell
        # for
        with pytest.raises(AssertionError):
            MarketMaker(buy_prices=self.buy_prices, environment=self.env,
                        sell_prices=self.buy_prices - 1)

    def test_energy_sold(self):
        # Test to check if the market maker properly cumulates all the energy it sells
        # and buys
        # Reset the market_maker to be sure there is no data present
        self.scenario.reset()
        market_maker = MarketMaker(buy_prices=self.buy_prices, environment=self.env)
        self.scenario.add_market(Market())
        assert len(market_maker.traded) == 0
        assert sum(market_maker.energy_sold) == 0
        assert sum(market_maker.energy_bought) == 0
        NR_TIME_STEPS = 10
        self.add_actor_w_constant_schedule("sell_actor", 1)
        self.run_simply(NR_TIME_STEPS)

        assert sum(market_maker.energy_bought) == 10

    def test_energy_bought(self):
        # Test to check if the market maker properly cumulates all the energy it sells
        # and buys
        # Reset the market_maker to be sure there is no data present
        self.scenario.reset()
        MarketMaker(buy_prices=self.buy_prices, environment=self.env)
        market_maker = self.env.market_maker
        self.scenario.add_market(Market())
        assert len(market_maker.traded) == 0
        assert sum(market_maker.energy_sold) == 0
        assert sum(market_maker.energy_bought) == 0
        NR_TIME_STEPS = 10
        self.add_actor_w_constant_schedule("buy_actor", -1)
        self.run_simply(NR_TIME_STEPS)
        assert sum(market_maker.energy_sold) == 10

    def run_simply(self, NR_TIME_STEPS):
        for _ in range(NR_TIME_STEPS):
            # actors calculate strategy based market interaction with the market maker
            self.scenario.create_strategies()

            # orders are generated based on the flexibility towards the planned market interaction
            # and a pricing scheme. Orders are matched at the end
            self.scenario.market_step()

            # actors are prepared for the next time step by changing socs, banks and predictions
            self.scenario.next_time_step()

    def add_actor_w_constant_schedule(self, name, schedule_value):
        actor = create_random(name)
        actor.data.load[:] = 0 + (schedule_value < 0) * abs(schedule_value)
        actor.data.schedule[:] = schedule_value
        actor.data.pv[:] = 0 + (schedule_value > 0) * schedule_value
        actor.battery.soc = 0
        # Adds actor to scenario, sets the environment and creates a prediction based on the
        # environment timestamp
        self.scenario.add_participant(actor)
        actor.create_prediction()

    def test_order_generation(self):
        self.scenario.reset()
        env = self.env
        grid_fee = 0.5
        cfg.config.default_grid_fee = grid_fee
        market_maker = MarketMaker(buy_prices=self.buy_prices, environment=self.env)

        ask_order, bid_order = self.generate_mm_order(market_maker)
        # Is the market_maker using the correct data
        assert self.buy_prices[env.time_step] == market_maker.all_sell_prices[env.time_step]
        assert self.buy_prices[env.time_step] == market_maker.all_sell_prices[env.time_step]

        # Do the orders use the correct market_maker data ?
        assert bid_order.price == market_maker.all_buy_prices[env.time_step]
        assert ask_order.price == market_maker.all_sell_prices[env.time_step]

        # test grid fee adjustment
        actor = create_random("test_actor")
        self.scenario.add_participant(actor)
        # Actor accessed the market maker prices which differ due to the grid_fee
        assert actor.get_mm_buy_prices()[0] == bid_order.price - grid_fee
        assert actor.get_mm_sell_prices()[0] == bid_order.price + grid_fee

        # test if the energy amount is correct
        assert bid_order.energy == MARKET_MAKER_THRESHOLD
        assert ask_order.energy == MARKET_MAKER_THRESHOLD

        # test if it works for different time steps
        env.time_step = 5
        market_maker.create_prediction()
        ask_order, bid_order = self.generate_mm_order(market_maker)
        assert bid_order.price == self.buy_prices[env.time_step]
        assert ask_order.price == self.buy_prices[env.time_step]

        assert actor.get_mm_buy_prices()[0] == bid_order.price - grid_fee
        assert actor.get_mm_sell_prices()[0] == ask_order.price + grid_fee

        # if no new prediction is created this should fail, in other words
        # forgetting to update with: market_maker.create_prediction()
        env.time_step = 7
        ask_order, bid_order = self.generate_mm_order(market_maker)
        with pytest.raises(AssertionError):
            assert bid_order.price == self.buy_prices[env.time_step]
        with pytest.raises(AssertionError):
            assert ask_order.price == self.buy_prices[env.time_step]

        # test use of sell prices
        # if both options of data and function are given, data is used
        kwarg = [dict(sell_prices=self.buy_prices * 2),
                 dict(sell_prices=self.buy_prices * 2, buy_to_sell_function=lambda x: x + 1)]
        for kw in kwarg:
            market_maker = MarketMaker(buy_prices=self.buy_prices, environment=self.env, **kw)
            env.time_step = 0
            market_maker.create_prediction()
            ask_order, bid_order = self.generate_mm_order(market_maker)
            assert bid_order.price * 2 == ask_order.price

        # if only function is given, function is used
        kwarg = dict(buy_to_sell_function=lambda x: x + 1.5)
        market_maker = MarketMaker(buy_prices=self.buy_prices, environment=self.env, **kwarg)
        env.time_step = 0
        market_maker.create_prediction()
        ask_order, bid_order = self.generate_mm_order(market_maker)
        assert bid_order.price + 1.5 == ask_order.price

    def generate_mm_order(self, market_maker):
        orders = market_maker.generate_orders()
        bid_order, = [order for order in orders if order.type == BID]
        ask_order, = [order for order in orders if order.type == ASK]
        return ask_order, bid_order
