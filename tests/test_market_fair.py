import pandas as pd
from pathlib import Path
import numpy as np
import pytest
from collections import namedtuple

from simply import scenario, market, market_2pac, market_fair
from simply.util import summerize_actor_trading

Offer = namedtuple("Offer", ["id", "time", "price", "energy", "seller"])
Bid = namedtuple("Bid", ["id", "time", "price", "energy", "buyer"])
FLOATING_POINT_TOLERANCE = 0

class TestBestMarket:
    def test_market_fair(self):
        sc = scenario.load(Path('/test_scenario'))
        power_network = sc.power_network
        m = market_fair.BestMarket(0, power_network)
        # TODO adapt to d3a
        # 'type': 1.0 {"order_id": "id", "actor_id": "buyer", "price": "energy_rate", "energy": "energy"}
        # 'type': -1.0 {"order_id": "id", "actor_id": "seller", "price": "energy_rate"}
        # data = {
        #     "market1": {
        #         "bids": [
        #             {"id": 1, "buyer": "A", "energy_rate": 1, "energy": 10},
        #             {"id": 2, "buyer": "B", "energy_rate": 2, "energy": 15},
        #             {"id": 3, "buyer": "C", "energy_rate": 3, "energy": 20},
        #         ],
        #         "offers": [
        #             {"id": 4, "seller": "A", "energy_rate": 1 + FLOATING_POINT_TOLERANCE,
        #              "energy": 25},
        #             {"id": 5, "seller": "B", "energy_rate": 5, "energy": 30},
        #             {"id": 6, "seller": "C", "energy_rate": 2.4, "energy": 35},
        #         ],
        #     },
        #     "market2": {
        #         "bids": [
        #             {"id": 7, "buyer": "A", "energy_rate": 1.5, "energy": 40},
        #             {"id": 8, "buyer": "B", "energy_rate": 2, "energy": 45},
        #             {"id": 9, "buyer": "C", "energy_rate": 6, "energy": 50},
        #         ],
        #         "offers": [
        #             {"id": 10, "seller": "A", "energy_rate": 1, "energy": 55},
        #             {"id": 11, "seller": "B", "energy_rate": 1, "energy": 60},
        #             {"id": 12, "seller": "C", "energy_rate": 1, "energy": 65},
        #         ],
        #     }
        # }
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
        # TODO adapt to d3a
        # expected_trades = [
        #     {"market_id": "market1",
        #      "bids": [{"id": 3, "buyer": "C", "energy_rate": 3, "energy": 20}],
        #      "offers": [{"id": 4, "seller": "A", "energy_rate": 1.00001, "energy": 25}],
        #      "selected_energy": 20, "trade_rate": 3},
        #
        #     {"market_id": "market2",
        #      "bids": [{"id": 8, "buyer": "B", "energy_rate": 2, "energy": 45}],
        #      "offers": [{"id": 12, "seller": "C", "energy_rate": 1, "energy": 65}],
        #      "selected_energy": 45, "trade_rate": 2},
        #
        #     {"market_id": "market2",
        #      "bids": [{"id": 9, "buyer": "C", "energy_rate": 6, "energy": 50}],
        #      "offers": [{"id": 11, "seller": "B", "energy_rate": 1, "energy": 60}],
        #      "selected_energy": 50, "trade_rate": 6}]
        expected_trades = [
            {'time': 0, 'bid_actor': 8, 'ask_actor': 4, 'energy': 0.4, 'price': 0.6908924092940778, 'ask_id': 16,'bid_id': 3, 'cluster': 2},
            {'time': 0, 'bid_actor': 1, 'ask_actor': 4, 'energy': 0.1, 'price': 0.2908924092940778,'ask_id': 11, 'bid_id': 0, 'cluster': 1},
            {'time': 0, 'bid_actor': 10, 'ask_actor': 4, 'energy': 0.1, 'price': 0.2908924092940778,'ask_id': 15, 'bid_id': 7, 'cluster': 1},
            {'time': 0, 'bid_actor': 7, 'ask_actor': 3, 'energy': 0.2, 'price': 0.41665171884290503,'ask_id': 8, 'bid_id': 1, 'cluster': 4}
        ]

        assert trades == expected_trades
        print(trades)

    def test_market_fair(self):
        sc = scenario.load(Path('../scenarios/test_scenario'))
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

        expected_trades = [
            {'time': 0, 'bid_actor': 8, 'ask_actor': 4, 'energy': 0.4, 'price': 0.6908924092940778, 'ask_id': 16,'bid_id': 3, 'cluster': 2},
            {'time': 0, 'bid_actor': 1, 'ask_actor': 4, 'energy': 0.1, 'price': 0.2908924092940778,'ask_id': 11, 'bid_id': 0, 'cluster': 1},
            {'time': 0, 'bid_actor': 10, 'ask_actor': 4, 'energy': 0.1, 'price': 0.2908924092940778,'ask_id': 15, 'bid_id': 7, 'cluster': 1},
            {'time': 0, 'bid_actor': 7, 'ask_actor': 3, 'energy': 0.2, 'price': 0.41665171884290503,'ask_id': 8, 'bid_id': 1, 'cluster': 4}
        ]

        assert trades == expected_trades
        print(trades)

