import numpy as np
import pytest

from simply.market import ASK, BID, MARKET_MAKER_THRESHOLD
from simply.market_maker import MarketMaker
import simply.config as cfg
from simply.scenario import Scenario
from simply.actor import create_random


class TestMarketMaker:
    cfg.Config("")
    buy_prices = np.arange(1, 100, 1)
    scenario = Scenario(None, [], None, buy_prices)
    env = scenario.environment

    def test_init(self):
        MarketMaker(environment=self.env, buy_prices=self.buy_prices)

    def test_price_comparison(self):
        MarketMaker(environment=self.env, buy_prices=self.buy_prices)
        # Assertion error should be thrown since mm would buy for higher prices than he would sell
        # for
        with pytest.raises(AssertionError):
            MarketMaker(environment=self.env, buy_prices=self.buy_prices,
                        sell_prices=self.buy_prices - 1)

    def test_order_generation(self):
        time_step = self.env.time_step
        grid_fee = 0.5
        cfg.config.default_grid_fee = grid_fee
        market_maker = MarketMaker(environment=self.env, buy_prices=self.buy_prices)
        orders = market_maker.generate_orders()
        bid_order, = [order for order in orders if order.type == BID]
        ask_order, = [order for order in orders if order.type == ASK]
        assert bid_order.price == self.buy_prices[time_step]
        assert ask_order.price == self.buy_prices[time_step]

        actor = create_random("test_actor", environment=self.env)
        assert actor.get_mm_buy_prices()[time_step] == bid_order.price - grid_fee
        assert actor.get_mm_sell_prices()[time_step] == bid_order.price + grid_fee

        # test if the energy amount is correct
        assert bid_order.energy == MARKET_MAKER_THRESHOLD
        assert ask_order.energy == MARKET_MAKER_THRESHOLD

        # test if it works for different time steps
        time_step = 5
        self.env.time_step = time_step
        market_maker.create_prediction()
        orders = market_maker.generate_orders()
        bid_order, = [order for order in orders if order.type == BID]
        ask_order, = [order for order in orders if order.type == ASK]
        assert bid_order.price == self.buy_prices[time_step]
        assert ask_order.price == self.buy_prices[time_step]

        assert actor.get_mm_buy_prices()[time_step] == bid_order.price - grid_fee
        assert actor.get_mm_sell_prices()[time_step] == bid_order.price + grid_fee

        # if no new prediction is created this should fail
        time_step = 7
        self.env.time_step = time_step
        orders = market_maker.generate_orders()
        bid_order, = [order for order in orders if order.type == BID]
        ask_order, = [order for order in orders if order.type == ASK]
        with pytest.raises(AssertionError):
            assert bid_order.price == self.buy_prices[time_step]
        with pytest.raises(AssertionError):
            assert ask_order.price == self.buy_prices[time_step] + grid_fee

        # test use of sell prices
        kwarg = dict(sell_prices=self.buy_prices * 2)
        market_maker = MarketMaker(environment=self.env, buy_prices=self.buy_prices, **kwarg)
        time_step = 0
        self.env.time_step = time_step
        market_maker.create_prediction()
        orders = market_maker.generate_orders()
        bid_order, = [order for order in orders if order.type == BID]
        ask_order, = [order for order in orders if order.type == ASK]
        assert bid_order.price * 2 == ask_order.price

        # if both options of data and function are given, data is used
        kwarg = dict(sell_prices=self.buy_prices * 2, buy_to_sell_function=lambda x: x + 1)
        market_maker = MarketMaker(environment=self.env, buy_prices=self.buy_prices, **kwarg)
        time_step = 0
        self.env.time_step = time_step
        market_maker.create_prediction()
        orders = market_maker.generate_orders()
        bid_order, = [order for order in orders if order.type == BID]
        ask_order, = [order for order in orders if order.type == ASK]
        assert bid_order.price * 2 == ask_order.price

        # if only function is given, function is used
        kwarg = dict(buy_to_sell_function=lambda x: x + 1.5)
        market_maker = MarketMaker(environment=self.env, buy_prices=self.buy_prices, **kwarg)
        time_step = 0
        self.env.time_step = time_step
        market_maker.create_prediction()
        orders = market_maker.generate_orders()
        bid_order, = [order for order in orders if order.type == BID]
        ask_order, = [order for order in orders if order.type == ASK]
        assert bid_order.price + 1.5 == ask_order.price
