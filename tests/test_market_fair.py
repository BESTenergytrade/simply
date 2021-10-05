from pathlib import Path
import pytest
from collections import namedtuple

from simply import scenario, market, market_2pac, market_fair

Offer = namedtuple("Offer", ["id", "time", "price", "energy", "seller"])
Bid = namedtuple("Bid", ["id", "time", "price", "energy", "buyer"])
FLOATING_POINT_TOLERANCE = 0

class TestBestMarket:
    def test_market_fair_none(self):
        sc = scenario.load(Path('test_scenario'))
        power_network = sc.power_network
        m = market_fair.BestMarket(0, power_network)

        data = None
        trades = m.match(data)
        assert trades == []

    def test_market_fair_no_orders(self):
        sc = scenario.load(Path('test_scenario'))
        power_network = sc.power_network
        m = market_fair.BestMarket(0, power_network)

        data = m.get_order_dict()
        trades = m.match(data)
        assert trades == []

    def test_market_fair_d3a_compatibility(self):
        pass

    def test_market_fair(self):
        sc = scenario.load(Path('test_scenario'))
        power_network = sc.power_network
        m = market_fair.BestMarket(0, power_network)
        # data = m.get_order_dict()
        data = {'market_1': {'bids': [
            {'type': 1.0, 'time': 8.0, 'actor_id': 1.0, 'energy': 0.1, 'price': 1.0221523639942367, 'order_id': 1.0},
            {'type': 1.0, 'time': 8.0, 'actor_id': 7.0, 'energy': 0.1, 'price': 0.8215954492900555, 'order_id': 4.0},
            {'type': 1.0, 'time': 8.0, 'actor_id': 7.0, 'energy': 0.1, 'price': 0.8215954492900555, 'order_id': 4.0},
            {'type': 1.0, 'time': 8.0, 'actor_id': 8.0, 'energy': 0.1, 'price': 0.8202179391645102, 'order_id': 5.0},
            {'type': 1.0, 'time': 8.0, 'actor_id': 8.0, 'energy': 0.1, 'price': 0.8202179391645102, 'order_id': 5.0},
            {'type': 1.0, 'time': 8.0, 'actor_id': 8.0, 'energy': 0.1, 'price': 0.8202179391645102, 'order_id': 5.0},
            {'type': 1.0, 'time': 8.0, 'actor_id': 8.0, 'energy': 0.1, 'price': 0.8202179391645102, 'order_id': 5.0},
            {'type': 1.0, 'time': 8.0, 'actor_id': 10.0, 'energy': 0.1, 'price': 0.6818185352490891, 'order_id': 6.0},
            {'type': 1.0, 'time': 8.0, 'actor_id': 10.0, 'energy': 0.1, 'price': 0.6818185352490891, 'order_id': 6.0}],
                            'offers': [
            {'type': -1.0, 'time': 8.0, 'actor_id': 0.0, 'energy': 0.1, 'price': 0.14764087116657698, 'order_id': 0.0},
            {'type': -1.0, 'time': 8.0, 'actor_id': 0.0, 'energy': 0.1, 'price': 0.14764087116657698, 'order_id': 0.0},
            {'type': -1.0, 'time': 8.0, 'actor_id': 0.0, 'energy': 0.1, 'price': 0.14764087116657698, 'order_id': 0.0},
            {'type': -1.0, 'time': 8.0, 'actor_id': 0.0, 'energy': 0.1, 'price': 0.14764087116657698, 'order_id': 0.0},
            {'type': -1.0, 'time': 8.0, 'actor_id': 0.0, 'energy': 0.1, 'price': 0.14764087116657698, 'order_id': 0.0},
            {'type': -1.0, 'time': 8.0, 'actor_id': 0.0, 'energy': 0.1, 'price': 0.14764087116657698, 'order_id': 0.0},
            {'type': -1.0, 'time': 8.0, 'actor_id': 3.0, 'energy': 0.1, 'price': 0.41665171884290503, 'order_id': 2.0},
            {'type': -1.0, 'time': 8.0, 'actor_id': 3.0, 'energy': 0.1, 'price': 0.41665171884290503, 'order_id': 2.0},
            {'type': -1.0, 'time': 8.0, 'actor_id': 3.0, 'energy': 0.1, 'price': 0.41665171884290503, 'order_id': 2.0},
            {'type': -1.0, 'time': 8.0, 'actor_id': 3.0, 'energy': 0.1, 'price': 0.41665171884290503, 'order_id': 2.0},
            {'type': -1.0, 'time': 8.0, 'actor_id': 3.0, 'energy': 0.1, 'price': 0.41665171884290503, 'order_id': 2.0},
            {'type': -1.0, 'time': 8.0, 'actor_id': 4.0, 'energy': 0.1, 'price': 0.2908924092940778, 'order_id': 3.0},
            {'type': -1.0, 'time': 8.0, 'actor_id': 4.0, 'energy': 0.1, 'price': 0.2908924092940778, 'order_id': 3.0},
            {'type': -1.0, 'time': 8.0, 'actor_id': 4.0, 'energy': 0.1, 'price': 0.2908924092940778, 'order_id': 3.0},
            {'type': -1.0, 'time': 8.0, 'actor_id': 4.0, 'energy': 0.1, 'price': 0.2908924092940778, 'order_id': 3.0},
            {'type': -1.0, 'time': 8.0, 'actor_id': 4.0, 'energy': 0.1, 'price': 0.2908924092940778, 'order_id': 3.0},
            {'type': -1.0, 'time': 8.0, 'actor_id': 4.0, 'energy': 0.1, 'price': 0.2908924092940778, 'order_id': 3.0}]
        }}

        data = {'market_1': {
            'bids': [
                {'actor_id': 1, 'energy': 0.1, 'price': 1, 'order_id': 1},
                {'actor_id': 7, 'energy': 0.1, 'price': 0.9, 'order_id': 4},
                {'actor_id': 7, 'energy': 0.1, 'price': 0.9, 'order_id': 4},
                {'actor_id': 8, 'energy': 0.1, 'price': 0.8, 'order_id': 5},
                {'actor_id': 8, 'energy': 0.1, 'price': 0.8, 'order_id': 5},
                {'actor_id': 8, 'energy': 0.1, 'price': 0.8, 'order_id': 5},
                {'actor_id': 8, 'energy': 0.1, 'price': 0.8, 'order_id': 5},
                {'actor_id': 10, 'energy': 0.1, 'price': .7, 'order_id': 6},
                {'actor_id': 10, 'energy': 0.1, 'price': .7, 'order_id': 6},
            ],
            'offers': [
                {'actor_id': 0, 'energy': 0.1, 'price': 0.1, 'order_id': 0},
                {'actor_id': 3, 'energy': 0.1, 'price': 0.4, 'order_id': 2},
                {'actor_id': 3, 'energy': 0.1, 'price': 0.4, 'order_id': 2},
                {'actor_id': 3, 'energy': 0.1, 'price': 0.4, 'order_id': 2},
                {'actor_id': 3, 'energy': 0.1, 'price': 0.4, 'order_id': 2},
                {'actor_id': 4, 'energy': 0.1, 'price': 0.3, 'order_id': 3},
                {'actor_id': 4, 'energy': 0.1, 'price': 0.3, 'order_id': 3},
                {'actor_id': 4, 'energy': 0.1, 'price': 0.3, 'order_id': 3},
            ]
        }}

        trades = m.match(data, False)

        # TODO: Example with unbalanced matching error -> replace by expected
        expected_trades = {
            1: {0: {'energy': 0.1, 'price': 1.0}},
            7: {3: {'energy': 0.2, 'price': 0.4}},
            8: {4: {'energy': 0.3, 'price': 0.7}},

        }

        assert len(trades) == sum(len(v) for v in expected_trades.values())

        for t in trades:
            print(t)
            bid_actor = t['bid_actor']
            ask_actor = t['ask_actor']
            energy = t['energy']
            price = t['price']
            assert bid_actor in expected_trades
            assert ask_actor in expected_trades[bid_actor]
            assert abs(energy - expected_trades[bid_actor][ask_actor]['energy']) < 1e-10
            assert abs(price - expected_trades[bid_actor][ask_actor]['price']) < 1e-10
