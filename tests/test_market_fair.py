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

        trades = m.match(data)

        # TODO: Example with unbalanced matching error -> replace by expected
        expected_trades = [
            {'time': 0, 'bid_actor': 8, 'ask_actor': 4, 'energy': 0.4, 'price': 0.6908924092940778, 'ask_id': 16,'bid_id': 3, 'cluster': 2},
            {'time': 0, 'bid_actor': 1, 'ask_actor': 4, 'energy': 0.1, 'price': 0.2908924092940778,'ask_id': 11, 'bid_id': 0, 'cluster': 1},
            {'time': 0, 'bid_actor': 10, 'ask_actor': 4, 'energy': 0.1, 'price': 0.2908924092940778,'ask_id': 15, 'bid_id': 7, 'cluster': 1},
            {'time': 0, 'bid_actor': 7, 'ask_actor': 3, 'energy': 0.2, 'price': 0.41665171884290503,'ask_id': 8, 'bid_id': 1, 'cluster': 4}
        ]

        assert trades == expected_trades
        print(trades)
