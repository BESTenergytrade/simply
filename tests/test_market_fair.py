from simply.actor import Order
from simply.market_fair import BestMarket, MARKET_MAKER_THRESHOLD, LARGE_ORDER_THRESHOLD
from simply.power_network import PowerNetwork
import simply.config as cfg
from simply.scenario import Scenario
from multidigraph import plot_agg_flows

import networkx as nx
import pytest
import pandas as pd


class TestBestMarket:
    cfg.Config("", "")
    cfg.config.energy_unit = 0.1
    nw = nx.Graph()
    nw.add_edges_from([(0, 1, {"weight": 1}), (1, 2), (1, 3), (0, 4)])
    pn = PowerNetwork("", nw, weight_factor=1)
    scenario = Scenario(None, None, [])

    def test_basic(self):
        """Tests the basic functionality of the BestMarket object to accept bids and asks via the
        accept_order method and correctly match asks and bids when the match method is called."""
        m = BestMarket(time_step=0, network=self.pn)
        # no orders: no matches
        matches = m.match()
        assert len(matches) == 0

        # only one type: no match
        m.accept_order(Order(-1, 0, 2, None, 1, 1))
        matches = m.match()
        assert len(matches) == 0

        # bid and ask with same energy and price
        m.accept_order(Order(1, 0, 3, None, 1, 1))
        matches = m.match()
        assert len(matches) == 1
        # check match
        assert matches[0]["time"] == 0
        assert matches[0]["bid_actor"] == 2
        assert matches[0]["ask_actor"] == 3
        assert matches[0]["energy"] == pytest.approx(1)
        assert matches[0]["price"] == pytest.approx(1)

    def test_prices_network(self):
        """Tests that the prices of the orders are correctly affected by the weights of
        the PowerNetwork."""
        # test prices with a given power network
        m = BestMarket(time_step=0, network=self.pn, disputed_matching="bid_price")
        # ask above bid: no match
        m.accept_order(Order(-1, 0, 2, None, 1, 2))
        m.accept_order(Order(1, 0, 3, None, 1, 2.5))
        matches = m.match()
        assert len(matches) == 0

        # reset orders
        m.orders = m.orders[:0]
        # ask below bid: take highest asking price
        m.accept_order(Order(-1, 0, 2, None, 1, 2.5))
        m.accept_order(Order(1, 0, 3, None, 1, 2))
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["energy"] == pytest.approx(1)
        assert matches[0]["price"] == pytest.approx(2)

        m.orders = m.orders[:0]
        # weight between nodes too high
        m.accept_order(Order(-1, 0, 2, None, 1, 3))
        m.accept_order(Order(1, 0, 4, None, 1, 3))
        matches = m.match()
        assert len(matches) == 0

        m.orders = m.orders[:0]
        # weight between nodes low enough
        m.accept_order(Order(-1, 0, 4, None, 1, 3))
        m.accept_order(Order(1, 0, 2, None, 1, 2))
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["energy"] == pytest.approx(1)
        assert matches[0]["price"] == pytest.approx(3)  # 2 + weight(1)

        m.orders = m.orders[:0]
        # match different clusters, even though there are orders from same cluster
        m.accept_order(Order(1, 0, 2, None, 1, 2))
        m.accept_order(Order(-1, 0, 3, None, 1, 2))
        m.accept_order(Order(-1, 0, 4, None, 1, 4))
        # expected: match 2 and 4, even though 2 and 3 are in same cluster (worse conditions)
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["ask_actor"] == 2
        assert matches[0]["bid_actor"] == 4
        assert matches[0]["energy"] == pytest.approx(1)
        assert matches[0]["price"] == pytest.approx(3)  # 2 + weight(1)

        m.orders = m.orders[:0]
        # same price: favor local orders
        # example: adjusted price is 4, actors 2 and 3 are same cluster
        m.accept_order(Order(-1, 0, 2, None, 1, 5))
        m.accept_order(Order(1, 0, 3, None, 1, 4))
        m.accept_order(Order(1, 0, 4, None, 1, 3))
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["ask_actor"] == 3
        assert matches[0]["energy"] == pytest.approx(1)
        assert matches[0]["price"] == pytest.approx(4)

    def test_prices_matrix(self):
        # test prices with a given grid fee matrix
        # example: cost 1 for trade between clusters
        m = BestMarket(time_step=0, grid_fee_matrix=[[0, 1], [1, 0]])

        # ask above bid: no match
        m.accept_order(Order(-1, 0, 2, 0, 1, 2))
        m.accept_order(Order(1, 0, 3, 0, 1, 2.5))
        matches = m.match()
        assert len(matches) == 0

        # reset orders
        m.orders = m.orders[:0]
        # ask below bid: take highest asking price
        m.accept_order(Order(-1, 0, 2, 1, 1, 2.5))
        m.accept_order(Order(1, 0, 3, 1, 1, 2))
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["energy"] == pytest.approx(1)
        assert matches[0]["price"] == pytest.approx(2)

        m.orders = m.orders[:0]
        # grid-fee between nodes too high
        m.accept_order(Order(-1, 0, 2, 0, 1, 3))
        m.accept_order(Order(1, 0, 4, 1, 1, 3))
        matches = m.match()
        assert len(matches) == 0

        m.orders = m.orders[:0]
        # grid-fee between nodes low enough
        m.accept_order(Order(-1, 0, 4, 0, 1, 3))
        m.accept_order(Order(1, 0, 2, 1, 1, 2))
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["energy"] == pytest.approx(1)
        assert matches[0]["price"] == pytest.approx(3)  # 2 + 1

        m.orders = m.orders[:0]
        # match different clusters, even though there are orders from same cluster
        m.accept_order(Order(1, 0, 2, 0, 1, 2))
        m.accept_order(Order(-1, 0, 3, 0, 1, 2))
        m.accept_order(Order(-1, 0, 4, 1, 1, 4))
        # expected: match 2 and 4, even though 2 and 3 are in same cluster (worse conditions)
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["ask_actor"] == 2
        assert matches[0]["bid_actor"] == 4
        assert matches[0]["energy"] == pytest.approx(1)
        assert matches[0]["price"] == pytest.approx(3)  # 2 + 1

        m.orders = m.orders[:0]
        # same price: favor local orders
        # example: adjusted price is 4, actors 2 and 3 are same cluster
        m.accept_order(Order(-1, 0, 2, 1, 1, 5))
        m.accept_order(Order(1, 0, 3, 1, 1, 4))
        m.accept_order(Order(1, 0, 4, 0, 1, 3))
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["ask_actor"] == 3
        assert matches[0]["energy"] == pytest.approx(1)
        assert matches[0]["price"] == pytest.approx(4)

    def test_energy(self):
        """Tests that the amount of energy traded equals the maximum amount available that is
        less than or equal to the amount requested by the bid."""
        # different energies
        m = BestMarket(time_step=0, network=self.pn)
        m.accept_order(Order(-1, 0, 2, None, .1, 1))
        m.accept_order(Order(1, 0, 3, None, 1, 1))
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["energy"] == pytest.approx(0.1)

        m.orders = m.orders[:0]
        m.accept_order(Order(-1, 0, 2, None, 100, 1))
        m.accept_order(Order(1, 0, 3, None, .3, 1))
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["energy"] == pytest.approx(0.3)

    def test_setting_order_id(self):
        # Check if matched orders retain original ID
        m = BestMarket(time_step=0, network=self.pn)
        m.accept_order(Order(-1, 0, 2, None, .2, 1), "ID1")
        m.accept_order(Order(1, 0, 3, None, 1, 1), "ID2")
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["energy"] == pytest.approx(0.2)
        assert matches[0]["bid_id"] == "ID1"
        assert matches[0]["ask_id"] == "ID2"

    def test_setting_id_market_maker(self):
        # Check if matched orders retain original ID for selling or buying market makers
        m = BestMarket(time_step=0, network=self.pn)
        # Test asking market maker with order ID
        m.accept_order(Order(-1, 0, 2, None, .3, 1), "ID1")
        m.accept_order(Order(1, 0, 5, None, MARKET_MAKER_THRESHOLD, 1), "ID2")
        matches = m.match()
        print(matches)
        assert len(matches) == 1
        assert matches[0]["energy"] == pytest.approx(0.3)
        assert matches[0]["bid_id"] == "ID1"
        assert matches[0]["ask_id"] == "ID2"
        assert matches[0]['ask_cluster'] is None

        # Reset orders
        m.orders = m.orders[:0]
        # Test bidding market maker with order ID
        m.accept_order(Order(-1, 0, 5, None, MARKET_MAKER_THRESHOLD, 1), "ID3")
        m.accept_order(Order(1, 0, 3, None, .3, 1), "ID4")
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["energy"] == pytest.approx(0.3)
        assert matches[0]["bid_id"] == "ID3"
        assert matches[0]["ask_id"] == "ID4"
        # maket maker does not have a cluster associated
        assert matches[0]['bid_cluster'] is None

    def test_multiple(self):
        """Tests that matches can be made which require multiple asks to satisfy one bid or multiple
        bids to satisfy one ask."""
        # multiple bids to satisfy one ask
        m = BestMarket(time_step=0, network=self.pn, disputed_matching="grid_fee")
        m.accept_order(Order(-1, 0, 2, None, .1, 4))
        m.accept_order(Order(-1, 0, 3, None, 3, 3))
        m.accept_order(Order(1, 0, 4, None, 2, 1))
        matches = m.match()
        assert len(matches) == 2
        assert matches[0]["energy"] == pytest.approx(0.1)
        assert matches[1]["energy"] == pytest.approx(1.9)  # only 2 in ask

        # multiple asks to satisfy one bid
        m.orders = m.orders[:0]
        m.accept_order(Order(1, 0, 2, None, 1, 1))
        m.accept_order(Order(1, 0, 2, None, 2, 2))
        m.accept_order(Order(1, 0, 3, None, 3, 3))
        m.accept_order(Order(1, 0, 3, None, 5, 4))
        m.accept_order(Order(-1, 0, 4, None, 10, 5))
        matches = m.match()
        assert len(matches) == 4
        assert matches[0]["energy"] == pytest.approx(1)
        assert matches[1]["energy"] == pytest.approx(2)
        assert matches[2]["energy"] == pytest.approx(3)
        assert matches[3]["energy"] == pytest.approx(4)  # only 100 in bid

    def test_match_ordering(self):
        """Test to check that matching favors local orders in case of equal (adjusted) price."""
        m = BestMarket(time_step=0, network=self.pn)
        m.accept_order(Order(-1, 0, 2, None, 1, 4))
        m.accept_order(Order(1, 0, 3, None, 1, 4))
        m.accept_order(Order(1, 0, 4, None, 1, 3))
        matches = m.match()
        # match cluster must be closest to bid cluster
        assert matches[0]['bid_cluster'] == 1
        assert matches[0]['ask_cluster'] == 1

        # test across multiple clusters
        lnw = nx.Graph()
        lnw.add_edges_from([(0, 1, {"weight": 1}), (1, 2), (1, 3), (0, 4),
                            (2, 5, {"weight": 1}), (5, 6)])
        lpn = PowerNetwork("", lnw)
        m = BestMarket(time_step=0, network=lpn)
        m.accept_order(Order(-1, 0, 6, None, 1, 5))
        m.accept_order(Order(1, 0, 3, None, 1, 4))
        m.accept_order(Order(1, 0, 4, None, 1, 3))
        matches = m.match()
        # match cluster must be closest to bid cluster
        assert matches[0]['bid_cluster'] == 2
        assert matches[0]['ask_cluster'] == 1

        # test that match doesn't prioritise local with price differential
        m = BestMarket(time_step=0, network=lpn)
        m.accept_order(Order(-1, 0, 6, None, 1, 100))
        m.accept_order(Order(1, 0, 3, None, 1, 50))
        m.accept_order(Order(1, 0, 4, None, 1, 3))
        matches = m.match()
        # match cluster must be closest to bid cluster
        assert matches[0]['price'] == 5

    def test_filter_large_orders(self):
        """Test to check that very large orders are ignored."""
        m = BestMarket(time_step=0, network=self.pn)
        m.accept_order(Order(-1, 0, 2, None, 1, 4))
        m.accept_order(Order(1, 0, 3, None, LARGE_ORDER_THRESHOLD + 1, 4))
        matches = m.match()
        # large ask is discarded, no match possible
        assert len(matches) == 0

    def test_market_maker_orders(self):
        """Test to check that market maker orders are not being ignored."""
        m = BestMarket(time_step=0, network=self.pn)
        m.accept_order(Order(-1, 0, 2, None, 1, 4))
        m.accept_order(Order(1, 0, 3, None, MARKET_MAKER_THRESHOLD, 4))
        matches = m.match()
        # matched with market maker
        assert len(matches) == 1
        assert matches[0]['energy'] == pytest.approx(1)
        assert matches[0]['price'] == pytest.approx(4)

    def test_update_clearing_cluster(self):
        """Test the update of a cluster clearing price is correctly done when a better match with
        another cluster is found."""
        cfg.config.default_grid_fee = 0
        cfg.config.energy_unit = 0.1
        m = BestMarket(time_step=0, network=self.pn)

        # Case "seller_c1_5" matched within cluster
        # add bids
        m.accept_order(Order(-1, 0, "buyer_c1_0", 1, 0.1, 10))  # will match with "seller_c1_5"
        m.accept_order(Order(-1, 0, "buyer_c1_1", 1, 0.1, 6))
        m.accept_order(Order(-1, 0, "buyer_c0_2", 0, 0.1, 10))  # could match with "seller_c1_5"
        m.accept_order(Order(-1, 0, "buyer_c0_3", 0, 0.1, 6))  # matching 2 energy units possible
        # add asks
        m.accept_order(Order(1, 0, "seller_c1_4", 1, 0.1, 6))
        m.accept_order(Order(1, 0, "seller_c1_5", 1, 0.1, 4))  # higher profit in Cluster 1
        m.accept_order(Order(1, 0, "seller_c1_6", 0, 0.1, 5.1))  # is price setting for Cluster 0
        matches = m.match()

        # seller_c1_5 (located in cluster 1) could match in
        # a) cluster 1 with buyer_c1_0 at clearing price 6 (due to price setting seller_c1_4)
        # b) cluster 0 with buyer_c0_0 at clearing price 5.1 (due to price setting seller_c0_6)
        #     adjusted_price is lower: 5 = 4 + 1 (due to additional fee)
        # TODO currently there is a bug that seller_c0_6 is only price setting as it cannot be
        #  matches in cluster 1 due to a low bid price of 6 < 5.1 + 1 (see tests below)
        # Assert that match a) exisits within matches
        assert any([match["ask_actor"] == "seller_c1_5" and match["bid_cluster"] == 1
                    and match["price"] == 6 for match in matches])

        # Case "seller_c1_5" matched in other cluster with more profit
        # reset order list
        m.orders = m.orders[:0]
        # add bids
        m.accept_order(Order(-1, 0, "buyer_c1_0", 1, 0.1, 10))
        m.accept_order(Order(-1, 0, "buyer_c1_1", 1, 0.1, 9))   # matching Cluster 1 asks possible
        m.accept_order(Order(-1, 0, "buyer_c0_2", 0, 0.1, 10))  # will match with "seller_c1_5"
        m.accept_order(Order(-1, 0, "buyer_c0_3", 0, 0.1, 9))  # matching 2 energy units
        # add asks
        m.accept_order(Order(1, 0, "seller_c1_4", 1, 0.1, 6))  # is price setting for Cluster 1
        m.accept_order(Order(1, 0, "seller_c1_5", 1, 0.1, 4))  # higher profit in other Cluster 0
        m.accept_order(Order(1, 0, "seller_c1_6", 0, 0.1, 7.1))  # is price setting for Cluster 0
        matches = m.match()

        match_in_more_profitable_cluster = {
            'time': 0,
            'bid_id': 2, 'ask_id': 5,
            'bid_actor': 'buyer_c0_2', 'ask_actor': 'seller_c1_5',
            'bid_cluster': 0, 'ask_cluster': 1,
            'energy': 0.1,
            'price': 7.1, 'included_grid_fee': 1
        }
        assert match_in_more_profitable_cluster in matches

        matched_energy = sum([match["energy"] for match in matches])
        assert matched_energy == pytest.approx(0.3)

    def test_update_clearing_cluster_bug1(self):
        """Test the update of a cluster clearing price is correctly done when a better match with
        another cluster is found."""
        cfg.config.default_grid_fee = 0
        cfg.config.energy_unit = 0.1
        m = BestMarket(time_step=0, network=self.pn, disputed_matching="grid_fee")

        # Case "seller_c1_5" matched within cluster
        # add bids
        m.accept_order(Order(-1, 0, "buyer_c1_0", 1, 0.1, 10))  # will match with "seller_c1_5"
        m.accept_order(Order(-1, 0, "buyer_c1_1", 1, 0.1, 9))
        m.accept_order(Order(-1, 0, "buyer_c0_2", 0, 0.1, 10))  # could match with "seller_c1_5"
        m.accept_order(Order(-1, 0, "buyer_c0_3", 0, 0.1, 9))  # matching 2 energy units possible
        # add asks
        m.accept_order(Order(1, 0, "seller_c1_4", 1, 0.1, 6))
        m.accept_order(Order(1, 0, "seller_c1_5", 1, 0.1, 4))  # higher profit in Cluster 1
        # TODO seller_c1_6 could achieve a higher profit in Cluster 1 leads to seller_c1_5 not
        #  matching in Cluster 1 but not at all
        m.accept_order(Order(1, 0, "seller_c1_6", 0, 0.1, 5))  # is price setting for Cluster 0
        matches = m.match()
        for match in matches:
            print(match)
        # seller_c1_5 could match in
        # a) cluster 1 with buyer_c1_0 at clearing price 6 (due to price setting seller_c1_4)
        # b) cluster 0 with buyer_c0_0 at clearing price 5 = 4 + 1 (due to additional fee)
        # Assert that match a) exisits within matches
        assert any([match["ask_actor"] == "seller_c1_5" and match["bid_cluster"] == 1
                    and match["price"] == 6 for match in matches])

    def test_update_clearing_cluster_bug2_matched_twice(self):
        """Test the update of a cluster clearing price is correctly done when a better match with
        another cluster is found."""
        cfg.config.default_grid_fee = 0
        cfg.config.energy_unit = 0.1
        m = BestMarket(time_step=0, network=self.pn, disputed_matching="grid_fee")

        # Case "seller_c1_5" matched within cluster
        # add bids
        m.accept_order(Order(-1, 0, "buyer_c1_0", 1, 0.1, 10))  # will match with "seller_c1_5"
        m.accept_order(Order(-1, 0, "buyer_c1_1", 1, 0.1, 7))
        m.accept_order(Order(-1, 0, "buyer_c0_2", 0, 0.1, 10))  # could match with "seller_c1_5"
        m.accept_order(
            Order(-1, 0, "buyer_c0_3", 0, 0.1, 7))  # matching 2 energy units possible
        # add asks
        m.accept_order(Order(1, 0, "seller_c1_4", 1, 0.1, 6))
        m.accept_order(Order(1, 0, "seller_c1_5", 1, 0.1, 4))  # higher profit in Cluster 1
        m.accept_order(
            Order(1, 0, "seller_c1_6", 0, 0.1, 5.1))  # is price setting for Cluster 0
        matches = m.match()
        for match in matches:
            print(match)
        # seller_c1_5 could match in
        # a) cluster 1 with buyer_c1_0 at clearing price 6 (due to price setting seller_c1_4)
        # b) cluster 0 with buyer_c0_0 at clearing price 5 = 4 + 1 (due to additional fee)
        # Assert that match a) exisits within matches
        assert any([match["ask_actor"] == "seller_c1_5" and match["bid_cluster"] == 1
                    and match["price"] == 6 for match in matches])

    def test_update_clearing_cluster_issue220(self):
        """Test the update of a cluster clearing price is correctly done when a better match with
        another cluster is found."""
        cfg.config.default_grid_fee = 0.01
        cfg.config.energy_unit = 0.01
        pn = PowerNetwork("", self.nw, weight_factor=0.003)
        m = BestMarket(time_step=0, network=pn, disputed_matching="grid_fee")

        # Cluster None: _MM
        # Cluster 0: Actor _1
        # Cluster 1: Actor _2, _3
        # ---------- add bids ----------------
        m.accept_order(Order(-1, 0, "buyer_MM", None, MARKET_MAKER_THRESHOLD, 0.04))
        # buyer_c0_1:
        # - after match with seller_c1_2 (0.2)
        # - the rest can be matched with seller_MM (0.17)
        m.accept_order(Order(-1, 0, "buyer_c0_1", 0, 0.37, 0.05))
        m.accept_order(Order(-1, 0, "buyer_c1_3", 1, 0.01, 0.05))
        # ---------- add asks ----------------
        m.accept_order(Order(1, 0, "seller_MM", None, MARKET_MAKER_THRESHOLD, 0.04))
        # should be matched in
        # - own cluster 1 (with buyer_c1_3) 0.01
        #   (even if profit is temporarily higher in cluster 0, as "best Cluster"
        # - other cluster 0 (with buyer_c0_1) 0.2, i.e. rest
        m.accept_order(Order(1, 0, "seller_c1_2", 1, 0.21, 0.03))
        print("\n")
        print(m.orders)
        assert m.get_grid_fee(bid_cluster=0, ask_cluster=1) == 0.003
        matches = m.match()
        print("\n")
        matches_df = pd.DataFrame(matches)
        print(matches_df.to_string())

        expected = [
            # {
            #     'time': 0,
            #     'bid_id': 2, 'ask_id': 4,
            #     'bid_actor': 'buyer_c1_3', 'ask_actor': 'seller_c1_2',
            #     'bid_cluster': 1, 'ask_cluster': 1,
            #     'energy': 0.01,
            #     'price': 0.03, 'included_grid_fee': 0.0
            # },
            {
                'time': 0,
                'bid_id': 2, 'ask_id': 3,
                'bid_actor': 'buyer_c1_3', 'ask_actor': 'seller_MM',
                'bid_cluster': 1, 'ask_cluster': None,
                'energy': 0.01,
                'price': 0.05, 'included_grid_fee': 0.01
            },
            {
                'time': 0,
                'bid_id': 1, 'ask_id': 4,
                'bid_actor': 'buyer_c0_1', 'ask_actor': 'seller_c1_2',
                'bid_cluster': 0, 'ask_cluster': 1,
                'energy': 0.21,
                'price': 0.05, 'included_grid_fee': 0.003
            },
            {
                'time': 0,
                'bid_id': 1, 'ask_id': 3,
                'bid_actor': 'buyer_c0_1', 'ask_actor': 'seller_MM',
                'bid_cluster': 0, 'ask_cluster': None,
                'energy': 0.16,
                'price': 0.05, 'included_grid_fee': 0.01
            }
        ]
        found_match = [False] * len(expected)
        for m in matches:
            for i, m1 in enumerate(expected):
                if m["bid_actor"] == m1["bid_actor"] and m["ask_actor"] == m1["ask_actor"]:
                    assert m["energy"] == pytest.approx(m1["energy"]), m
                    found_match[i] = True
        assert all(found_match)

    def test_update_clearing_cluster_issue220_b(self):
        """Test the update of a cluster clearing price is correctly done when a better match with
        another cluster is found."""
        cfg.config.default_grid_fee = 0.01
        cfg.config.energy_unit = 0.01
        pn = PowerNetwork("", self.nw, weight_factor=0.003)
        m = BestMarket(time_step=0, network=pn, disputed_matching="grid_fee")

        # Cluster None: _MM
        # Cluster 0: Actor _1
        # Cluster 1: Actor _2, _3
        # ---------- add bids ----------------
        m.accept_order(Order(-1, 0, "buyer_MM", None, MARKET_MAKER_THRESHOLD, 0.04))
        # buyer_c0_1:
        # - after match with seller_c1_2 (0.2)
        # - the rest can be matched with seller_MM (0.17)
        m.accept_order(Order(-1, 0, "buyer_c0_1", 0, 0.37, 0.05))
        m.accept_order(Order(-1, 0, "buyer_c1_3", 1, 0.07, 0.05))
        # ---------- add asks ----------------
        m.accept_order(Order(1, 0, "seller_MM", None, MARKET_MAKER_THRESHOLD, 0.04))
        # should be matched in
        # - own cluster 1 (with buyer_c1_3) 0.07
        #   (even if profit is temporarily higher in cluster 0, as "best Cluster"
        # - other cluster 0 (with buyer_c0_1) 0.20, i.e. rest
        m.accept_order(Order(1, 0, "seller_c1_2", 1, 0.27, 0.03))

        print("\n")
        print(m.orders)
        assert m.get_grid_fee(bid_cluster=0, ask_cluster=1) == 0.003
        matches = m.match()
        print("\n")
        matches_df = pd.DataFrame(matches)
        print(matches_df.to_string())

        expected = [
            {
                'time': 0,
                'bid_id': 2, 'ask_id': 4,
                'bid_actor': 'buyer_c1_3', 'ask_actor': 'seller_c1_2',
                'bid_cluster': 1, 'ask_cluster': 1,
                'energy': 0.06,
                'price': 0.03, 'included_grid_fee': 0.0
            },
            {
                'time': 0,
                'bid_id': 1, 'ask_id': 4,
                'bid_actor': 'buyer_c0_1', 'ask_actor': 'seller_c1_2',
                'bid_cluster': 0, 'ask_cluster': 1,
                'energy': 0.21,
                'price': 0.05, 'included_grid_fee': 0.003
            },
            {
                'time': 0,
                'bid_id': 1, 'ask_id': 3,
                'bid_actor': 'buyer_c0_1', 'ask_actor': 'seller_MM',
                'bid_cluster': 0, 'ask_cluster': None,
                'energy': 0.16,
                'price': 0.05, 'included_grid_fee': 0.01
            },
            {
                'time': 0,
                'bid_id': 2, 'ask_id': 3,
                'bid_actor': 'buyer_c1_3', 'ask_actor': 'seller_MM',
                'bid_cluster': 1, 'ask_cluster': None,
                'energy': 0.01,
                'price': 0.05, 'included_grid_fee': 0.01
            }
        ]
        found_match = [False] * len(expected)
        for m in matches:
            for i, m1 in enumerate(expected):
                if m["bid_actor"] == m1["bid_actor"] and m["ask_actor"] == m1["ask_actor"]:
                    assert m["energy"] == pytest.approx(m1["energy"]), m
                    found_match[i] = True
        assert all(found_match)

    '''
    def test_update_clearing_cluster_issue220_c(self):
        """Test the update of a cluster clearing price is correctly done when a better match with
        another cluster is found.

        case: ask energy < bid energy

        # currently the ask with the lowest price gets a worse clearing price then the second lowest
        # TODO reorganize matched orders in final matching step
        """
        cfg.config.default_grid_fee = 0.1
        cfg.config.energy_unit = 0.01
        pn = PowerNetwork("", self.nw, weight_factor=0.03)
        m = BestMarket(time_step=0, network=pn, disputed_matching="grid_fee")

        # Cluster None: _MM
        # Cluster 0: Actor _1
        # Cluster 1: Actor _2, _3, _4
        # ---------- add bids ----------------
        m.accept_order(Order(-1, 0, "buyer_MM", None, MARKET_MAKER_THRESHOLD, 0.05))
        # buyer_c0_1:
        # - expected ...
        m.accept_order(Order(-1, 0, "buyer_c0_1", 0, 0.37, 0.15))
        m.accept_order(Order(-1, 0, "buyer_c1_3", 1, 0.07, 0.13))
        # ---------- add asks ----------------
        m.accept_order(Order(1, 0, "seller_MM", None, MARKET_MAKER_THRESHOLD, 0.04))
        # - expected ...
        m.accept_order(Order(1, 0, "seller_c1_2", 1, 0.27, 0.02))
        m.accept_order(Order(1, 0, "seller_c1_4", 1, 0.12, 0.04))

        print("\n")
        print(m.orders)
        assert m.get_grid_fee(bid_cluster=0, ask_cluster=1) == 0.03
        matches = m.match()
        print("\n")
        matches_df = pd.DataFrame(matches)
        print(matches_df.to_string())

        plot_agg_flows(matches_df)

        '''
          type time     actor_id cluster                 energy price
        0   -1    0     buyer_MM    None  9223372036854775808.0  0.05
        1   -1    0   buyer_c0_1       0                   0.37  0.15
        2   -1    0   buyer_c1_3       1                   0.07  0.13
        3    1    0    seller_MM    None  9223372036854775808.0  0.04
        4    1    0  seller_c1_2       1                   0.27  0.02
        5    1    0  seller_c1_4       1                   0.12  0.04
        
           time  bid_id  ask_id   bid_actor    ask_actor  bid_cluster  ask_cluster  energy  price  included_grid_fee
        0     0       1       4  buyer_c0_1  seller_c1_2            0          1.0    0.27   0.14               0.03
        1     0       1       5  buyer_c0_1  seller_c1_4            0          1.0    0.09   0.14               0.03
        2     0       2       5  buyer_c1_3  seller_c1_4            1          1.0    0.03   0.04               0.00
        3     0       1       3  buyer_c0_1    seller_MM            0          NaN    0.01   0.14               0.10
        '''

        assert False

    def test_update_clearing_cluster_issue220_new(self):
        """Test the update of a cluster clearing price is correctly done when a better match with
        another cluster is found."""
        cfg.config.default_grid_fee = 0.01
        cfg.config.energy_unit = 0.01
        pn = PowerNetwork("", self.nw, weight_factor=0.003)
        m = BestMarket(time_step=0, network=pn, disputed_matching="grid_fee")

        # Cluster None: _MM
        # Cluster 0: Actor _1
        # Cluster 1: Actor _2, _3, _4
        # ---------- add bids ----------------
        m.accept_order(Order(-1, 0, "buyer_MM", None, MARKET_MAKER_THRESHOLD, 0.05))
        # buyer_c0_1:
        # - expected ...
        m.accept_order(Order(-1, 0, "buyer_c0_1", 0, 0.1, 0.06))
        m.accept_order(Order(-1, 0, "buyer_c1_2", 1, 0.17, 0.06))
        # ---------- add asks ----------------
        m.accept_order(Order(1, 0, "seller_MM", None, MARKET_MAKER_THRESHOLD, 0.05))
        # - expected ...
        m.accept_order(Order(1, 0, "seller_c1_3", 1, 0.30, 0.04))

        print("\n")
        print(m.orders)
        assert m.get_grid_fee(bid_cluster=0, ask_cluster=1) == 0.003
        matches = m.match()
        print("\n")
        matches_df = pd.DataFrame(matches)
        print(matches_df.to_string())

        plot_agg_flows(matches_df)

        '''
        locally unmatched ask of seller_c1_3 is not sold to market maker
        - TM: yes this is an error, market maker bid cluster is not correctly cleared
        
          type time     actor_id cluster                 energy price
        0   -1    0     buyer_MM    None  9223372036854775808.0  0.05
        1   -1    0   buyer_c0_1       0                    0.1  0.06
        2   -1    0   buyer_c1_2       1                   0.17  0.06
        3    1    0    seller_MM    None  9223372036854775808.0  0.05
        4    1    0  seller_c1_3       1                    0.3  0.04
        
        
           time  bid_id  ask_id   bid_actor    ask_actor  bid_cluster  ask_cluster  energy  price  included_grid_fee
        0     0       2       4  buyer_c1_2  seller_c1_3            1            1    0.17   0.04                  0
        '''

        assert False

    def test_update_clearing_cluster_issue220_d(self):
        """Test a scenario without batteries and with load surplus. MarketMaker selling price is
        greater than MarketMaker buying price. """
        cfg.config.default_grid_fee = 0.01
        cfg.config.energy_unit = 0.01
        pn = PowerNetwork("", self.nw, weight_factor=0.003)
        m = BestMarket(time_step=0, network=pn, disputed_matching="grid_fee")

        # Cluster None: _MM
        # Cluster 0: Actor _1
        # Cluster 1: Actor _2, _3, _4
        # ---------- add bids ----------------
        m.accept_order(Order(-1, 0, "buyer_MM", None, MARKET_MAKER_THRESHOLD, 0.04))
        # buyer_c0_1:
        # - expected ...
        m.accept_order(Order(-1, 0, "buyer_c0_1", 0, 0.37, 0.06))
        m.accept_order(Order(-1, 0, "buyer_c1_3", 1, 0.07, 0.06))
        # ---------- add asks ----------------
        m.accept_order(Order(1, 0, "seller_MM", None, MARKET_MAKER_THRESHOLD, 0.05))
        # - expected ...
        m.accept_order(Order(1, 0, "seller_c1_2", 1, 0.20, 0.03))

        print("\n")
        print(m.orders)
        assert m.get_grid_fee(bid_cluster=0, ask_cluster=1) == 0.003
        matches = m.match()
        print("\n")
        matches_df = pd.DataFrame(matches)
        print(matches_df.to_string())

        # plot_agg_flows(matches_df)

        '''
          type time     actor_id cluster                 energy price
        0   -1    0     buyer_MM    None  9223372036854775808.0  0.04
        1   -1    0   buyer_c0_1       0                   0.37  0.06
        2   -1    0   buyer_c1_3       1                   0.07  0.06
        3    1    0    seller_MM    None  9223372036854775808.0  0.05
        4    1    0  seller_c1_2       1                    0.2  0.03
        
        
           time  bid_id  ask_id   bid_actor    ask_actor  bid_cluster  ask_cluster  energy  price  included_grid_fee
        0     0       1       4  buyer_c0_1  seller_c1_2            0            1    0.05  0.033              0.003
        1     0       2       4  buyer_c1_3  seller_c1_2            1            1    0.07  0.030              0.000
        '''

        assert False

    def test_update_clearing_cluster_issue220_e(self):
        """Test a scenario without batteries and with load surplus. MarketMaker selling price is
        equal to MarketMaker buying price.  """
        cfg.config.default_grid_fee = 0.01
        cfg.config.energy_unit = 0.01
        pn = PowerNetwork("", self.nw, weight_factor=0.003)
        m = BestMarket(time_step=0, network=pn, disputed_matching="grid_fee")

        # Cluster None: _MM
        # Cluster 0: Actor _1
        # Cluster 1: Actor _2, _3, _4
        # ---------- add bids ----------------
        m.accept_order(Order(-1, 0, "buyer_MM", None, MARKET_MAKER_THRESHOLD, 0.05))
        # buyer_c0_1:
        # - expected ...
        m.accept_order(Order(-1, 0, "buyer_c0_1", 0, 0.37, 0.06))
        m.accept_order(Order(-1, 0, "buyer_c1_3", 1, 0.07, 0.06))
        # ---------- add asks ----------------
        m.accept_order(Order(1, 0, "seller_MM", None, MARKET_MAKER_THRESHOLD, 0.05))
        # - expected ...
        m.accept_order(Order(1, 0, "seller_c1_2", 1, 0.20, 0.04))

        print("\n")
        print(m.orders)
        assert m.get_grid_fee(bid_cluster=0, ask_cluster=1) == 0.003
        matches = m.match()
        print("\n")
        matches_df = pd.DataFrame(matches)
        print(matches_df.to_string())

        # plot_agg_flows(matches_df)

        '''
         type time     actor_id cluster                 energy price
        0   -1    0     buyer_MM    None  9223372036854775808.0  0.05
        1   -1    0   buyer_c0_1       0                   0.37  0.06
        2   -1    0   buyer_c1_3       1                   0.07  0.06
        3    1    0    seller_MM    None  9223372036854775808.0  0.05
        4    1    0  seller_c1_2       1                    0.2  0.04
        
        
           time  bid_id  ask_id   bid_actor    ask_actor  bid_cluster  ask_cluster  energy  price  included_grid_fee
        0     0       1       4  buyer_c0_1  seller_c1_2            0            1    0.05  0.043              0.003
        1     0       2       4  buyer_c1_3  seller_c1_2            1            1    0.07  0.040              0.000
        '''

        assert False

    def test_update_clearing_cluster_issue220_f(self):
        """Test a scenario without batteries and with generation surplus. MarketMaker buying price is
        equal to MarketMaker selling price. """
        cfg.config.default_grid_fee = 0.01
        cfg.config.energy_unit = 0.01
        pn = PowerNetwork("", self.nw, weight_factor=0.003)
        m = BestMarket(time_step=0, network=pn, disputed_matching="grid_fee")

        # Cluster None: _MM
        # Cluster 0: Actor _1
        # Cluster 1: Actor _2, _3, _4
        # ---------- add bids ----------------
        m.accept_order(Order(-1, 0, "buyer_MM", None, MARKET_MAKER_THRESHOLD, 0.05))
        # buyer_c0_1:
        # - expected ...
        m.accept_order(Order(-1, 0, "buyer_c0_1", 0, 0.10, 0.06))
        m.accept_order(Order(-1, 0, "buyer_c1_3", 1, 0.07, 0.06))
        # ---------- add asks ----------------
        m.accept_order(Order(1, 0, "seller_MM", None, MARKET_MAKER_THRESHOLD, 0.05))
        # - expected ...
        m.accept_order(Order(1, 0, "seller_c1_2", 1, 0.27, 0.04))

        print("\n")
        print(m.orders)
        assert m.get_grid_fee(bid_cluster=0, ask_cluster=1) == 0.003
        matches = m.match()
        print("\n")
        matches_df = pd.DataFrame(matches)
        print(matches_df.to_string())

        # plot_agg_flows(matches_df)

        '''
          type time     actor_id cluster                 energy price
        0   -1    0     buyer_MM    None  9223372036854775808.0  0.05
        1   -1    0   buyer_c0_1       0                    0.1  0.06
        2   -1    0   buyer_c1_3       1                   0.07  0.06
        3    1    0    seller_MM    None  9223372036854775808.0  0.05
        4    1    0  seller_c1_2       1                   0.27  0.04
        
        
           time  bid_id  ask_id   bid_actor    ask_actor  bid_cluster  ask_cluster  energy  price  included_grid_fee
        0     0       1       4  buyer_c0_1  seller_c1_2          0.0            1    0.10  0.043              0.003
        1     0       2       4  buyer_c1_3  seller_c1_2          1.0            1    0.07  0.040              0.000
        2     0       0       4    buyer_MM  seller_c1_2          NaN            1    0.01  0.050              0.01010
        '''

        assert False

    def test_update_clearing_cluster_issue220_g(self):
        """Test a scenario with batteries and with load surplus. MarketMaker selling price is
        equal to MarketMaker buying price. Buying is urgent. Selling price of local actor is equal to
        selling price of MarketMaker"""
        cfg.config.default_grid_fee = 0.01
        cfg.config.energy_unit = 0.01
        pn = PowerNetwork("", self.nw, weight_factor=0.003)
        m = BestMarket(time_step=0, network=pn, disputed_matching="grid_fee")

        # Cluster None: _MM
        # Cluster 0: Actor _1
        # Cluster 1: Actor _2, _3, _4
        # ---------- add bids ----------------
        m.accept_order(Order(-1, 0, "buyer_MM", None, MARKET_MAKER_THRESHOLD, 0.05))
        # buyer_c0_1:
        # - expected ...
        m.accept_order(Order(-1, 0, "buyer_c0_1", 0, 0.37, 0.06))
        m.accept_order(Order(-1, 0, "buyer_c1_3", 1, 0.07, 0.06))
        # ---------- add asks ----------------
        m.accept_order(Order(1, 0, "seller_MM", None, MARKET_MAKER_THRESHOLD, 0.05))
        # - expected ...
        m.accept_order(Order(1, 0, "seller_c1_2", 1, 0.20, 0.05))

        print("\n")
        print(m.orders)
        assert m.get_grid_fee(bid_cluster=0, ask_cluster=1) == 0.003
        matches = m.match()
        print("\n")
        matches_df = pd.DataFrame(matches)
        print(matches_df.to_string())

        # plot_agg_flows(matches_df)

        '''
          type time     actor_id cluster                 energy price
        0   -1    0     buyer_MM    None  9223372036854775808.0  0.05
        1   -1    0   buyer_c0_1       0                   0.37  0.06
        2   -1    0   buyer_c1_3       1                   0.07  0.06
        3    1    0    seller_MM    None  9223372036854775808.0  0.05
        4    1    0  seller_c1_2       1                    0.2  0.05
        
        
           time  bid_id  ask_id   bid_actor    ask_actor  bid_cluster  ask_cluster  energy  price  included_grid_fee
        0     0       1       4  buyer_c0_1  seller_c1_2            0            1    0.12  0.053              0.003
        1     0       2       4  buyer_c1_3  seller_c1_2            1            1    0.07  0.050              0.000
        '''

        assert False

    def test_update_clearing_cluster_issue220_h(self):
        """Test a scenario with batteries and with load surplus. MarketMaker selling price is
        equal to MarketMaker buying price. Buying is urgent. Selling price of local actor is equal
        to selling price of MarketMaker + 0.5 * grid fee"""
        cfg.config.default_grid_fee = 0.01
        cfg.config.energy_unit = 0.01
        pn = PowerNetwork("", self.nw, weight_factor=0.003)
        m = BestMarket(time_step=0, network=pn, disputed_matching="grid_fee")

        # Cluster None: _MM
        # Cluster 0: Actor _1
        # Cluster 1: Actor _2, _3, _4
        # ---------- add bids ----------------
        m.accept_order(Order(-1, 0, "buyer_MM", None, MARKET_MAKER_THRESHOLD, 0.05))
        # buyer_c0_1:
        # - expected ...
        m.accept_order(Order(-1, 0, "buyer_c0_1", 0, 0.37, 0.06))
        m.accept_order(Order(-1, 0, "buyer_c1_3", 1, 0.07, 0.06))
        # ---------- add asks ----------------
        m.accept_order(Order(1, 0, "seller_MM", None, MARKET_MAKER_THRESHOLD, 0.05))
        # - expected ...
        m.accept_order(Order(1, 0, "seller_c1_2", 1, 0.20, 0.055))

        print("\n")
        print(m.orders)
        assert m.get_grid_fee(bid_cluster=0, ask_cluster=1) == 0.003
        matches = m.match()
        print("\n")
        matches_df = pd.DataFrame(matches)
        print(matches_df.to_string())

        # plot_agg_flows(matches_df)

        '''
        type time     actor_id cluster                 energy  price
        0   -1    0     buyer_MM    None  9223372036854775808.0   0.05
        1   -1    0   buyer_c0_1       0                   0.37   0.06
        2   -1    0   buyer_c1_3       1                   0.07   0.06
        3    1    0    seller_MM    None  9223372036854775808.0   0.05
        4    1    0  seller_c1_2       1                    0.2  0.055
        
        
           time  bid_id  ask_id   bid_actor    ask_actor  bid_cluster  ask_cluster  energy  price  included_grid_fee
        0     0       1       4  buyer_c0_1  seller_c1_2            0            1    0.12  0.058              0.003
        1     0       2       4  buyer_c1_3  seller_c1_2            1            1    0.07  0.055              0.000
        '''

        assert False

    def test_update_clearing_cluster_issue220_i(self):
        """Test a scenario with batteries and with load surplus. MarketMaker selling price is
        equal to MarketMaker buying price. Selling is urgent. Buying prices of local actors are less
        than selling price od MarketMaker and equal to or less than selling price of local actors"""
        cfg.config.default_grid_fee = 0.01
        cfg.config.energy_unit = 0.01
        pn = PowerNetwork("", self.nw, weight_factor=0.003)
        m = BestMarket(time_step=0, network=pn, disputed_matching="grid_fee")

        # Cluster None: _MM
        # Cluster 0: Actor _1
        # Cluster 1: Actor _2, _3, _4
        # ---------- add bids ----------------
        m.accept_order(Order(-1, 0, "buyer_MM", None, MARKET_MAKER_THRESHOLD, 0.05))
        # buyer_c0_1:
        # - expected ...
        m.accept_order(Order(-1, 0, "buyer_c0_1", 0, 0.37, 0.04))
        m.accept_order(Order(-1, 0, "buyer_c1_3", 1, 0.07, 0.03))
        # ---------- add asks ----------------
        m.accept_order(Order(1, 0, "seller_MM", None, MARKET_MAKER_THRESHOLD, 0.05))
        # - expected ...
        m.accept_order(Order(1, 0, "seller_c1_2", 1, 0.20, 0.04))

        print("\n")
        print(m.orders)
        assert m.get_grid_fee(bid_cluster=0, ask_cluster=1) == 0.003
        matches = m.match()
        print("\n")
        matches_df = pd.DataFrame(matches)
        print(matches_df.to_string())

        # plot_agg_flows(matches_df)

        '''
          type time     actor_id cluster                 energy price
        0   -1    0     buyer_MM    None  9223372036854775808.0  0.05
        1   -1    0   buyer_c0_1       0                   0.37  0.04
        2   -1    0   buyer_c1_3       1                   0.07  0.03
        3    1    0    seller_MM    None  9223372036854775808.0  0.05
        4    1    0  seller_c1_2       1                    0.2  0.04
        
        
           time  bid_id  ask_id bid_actor    ask_actor bid_cluster  ask_cluster  energy  price  included_grid_fee
        0     0       0       4  buyer_MM  seller_c1_2        None            1     0.2   0.05               0.01
        '''

        assert False
    '''

    def test_update_clearing_cluster_issue220_j(self):
        """Test a scenario with batteries and generation surplus. MarketMaker selling price is
        equal to MarketMaker buying price. Buying is urgent. Selling price of local actor is equal to
        selling price of MarketMaker"""
        cfg.config.default_grid_fee = 0.01
        cfg.config.energy_unit = 0.01
        pn = PowerNetwork("", self.nw, weight_factor=0.003)
        m = BestMarket(time_step=0, network=pn, disputed_matching="grid_fee")

        # Cluster None: _MM
        # Cluster 0: Actor _1
        # Cluster 1: Actor _2, _3, _4
        # ---------- add bids ----------------
        m.accept_order(Order(-1, 0, "buyer_MM", None, MARKET_MAKER_THRESHOLD, 0.05))
        # buyer_c0_1:
        # - expected ...
        m.accept_order(Order(-1, 0, "buyer_c0_1", 0, 0.10, 0.06))
        m.accept_order(Order(-1, 0, "buyer_c1_3", 1, 0.07, 0.06))
        # ---------- add asks ----------------
        m.accept_order(Order(1, 0, "seller_MM", None, MARKET_MAKER_THRESHOLD, 0.05))
        # - expected ...
        m.accept_order(Order(1, 0, "seller_c1_2", 1, 0.27, 0.05))

        print("\n")
        print(m.orders)
        assert m.get_grid_fee(bid_cluster=0, ask_cluster=1) == 0.003
        matches = m.match()
        print("\n")
        matches_df = pd.DataFrame(matches)
        print(matches_df.to_string())

        # plot_agg_flows(matches_df)

        '''
          type time     actor_id cluster                 energy price
        0   -1    0     buyer_MM    None  9223372036854775808.0  0.05
        1   -1    0   buyer_c0_1       0                    0.1  0.06
        2   -1    0   buyer_c1_3       1                   0.07  0.06
        3    1    0    seller_MM    None  9223372036854775808.0  0.05
        4    1    0  seller_c1_2       1                   0.27  0.05
        
        
           time  bid_id  ask_id   bid_actor    ask_actor  bid_cluster  ask_cluster  energy  price  included_grid_fee
        0     0       1       4  buyer_c0_1  seller_c1_2            0            1    0.10  0.053              0.003
        1     0       2       4  buyer_c1_3  seller_c1_2            1            1    0.07  0.050              0.000
        '''

        assert False

    def test_update_clearing_cluster_issue220_k(self):
        """Test a scenario with batteries and generation surplus. MarketMaker selling price is
        equal to MarketMaker buying price. Buying is urgent. Selling price of local actor is equal
        to selling price of MarketMaker + 0.5 * grid fee"""
        cfg.config.default_grid_fee = 0.01
        cfg.config.energy_unit = 0.01
        pn = PowerNetwork("", self.nw, weight_factor=0.003)
        m = BestMarket(time_step=0, network=pn, disputed_matching="grid_fee")

        # Cluster None: _MM
        # Cluster 0: Actor _1
        # Cluster 1: Actor _2, _3, _4
        # ---------- add bids ----------------
        m.accept_order(Order(-1, 0, "buyer_MM", None, MARKET_MAKER_THRESHOLD, 0.05))
        # buyer_c0_1:
        # - expected ...
        m.accept_order(Order(-1, 0, "buyer_c0_1", 0, 0.10, 0.06))
        m.accept_order(Order(-1, 0, "buyer_c1_3", 1, 0.07, 0.06))
        # ---------- add asks ----------------
        m.accept_order(Order(1, 0, "seller_MM", None, MARKET_MAKER_THRESHOLD, 0.05))
        # - expected ...
        m.accept_order(Order(1, 0, "seller_c1_2", 1, 0.27, 0.055))

        print("\n")
        print(m.orders)
        assert m.get_grid_fee(bid_cluster=0, ask_cluster=1) == 0.003
        matches = m.match()
        print("\n")
        matches_df = pd.DataFrame(matches)
        print(matches_df.to_string())

        # plot_agg_flows(matches_df)

        '''
          type time     actor_id cluster                 energy  price
        0   -1    0     buyer_MM    None  9223372036854775808.0   0.05
        1   -1    0   buyer_c0_1       0                    0.1   0.06
        2   -1    0   buyer_c1_3       1                   0.07   0.06
        3    1    0    seller_MM    None  9223372036854775808.0   0.05
        4    1    0  seller_c1_2       1                   0.27  0.055
        
        
           time  bid_id  ask_id   bid_actor    ask_actor  bid_cluster  ask_cluster  energy  price  included_grid_fee
        0     0       1       4  buyer_c0_1  seller_c1_2            0            1    0.10  0.058              0.003
        1     0       2       4  buyer_c1_3  seller_c1_2            1            1    0.07  0.055              0.000
        '''

        assert False

    def test_update_clearing_cluster_issue220_l(self):
        """Test a scenario with batteries and generation surplus. MarketMaker selling price is
        equal to MarketMaker buying price. Selling is urgent. Buying prices of local actors are less
        than selling price od MarketMaker and equal to or less than selling price of local actors"""
        cfg.config.default_grid_fee = 0.01
        cfg.config.energy_unit = 0.01
        pn = PowerNetwork("", self.nw, weight_factor=0.003)
        m = BestMarket(time_step=0, network=pn, disputed_matching="grid_fee")

        # Cluster None: _MM
        # Cluster 0: Actor _1
        # Cluster 1: Actor _2, _3, _4
        # ---------- add bids ----------------
        m.accept_order(Order(-1, 0, "buyer_MM", None, MARKET_MAKER_THRESHOLD, 0.05))
        # buyer_c0_1:
        # - expected ...
        m.accept_order(Order(-1, 0, "buyer_c0_1", 0, 0.10, 0.04))
        m.accept_order(Order(-1, 0, "buyer_c1_3", 1, 0.07, 0.03))
        # ---------- add asks ----------------
        m.accept_order(Order(1, 0, "seller_MM", None, MARKET_MAKER_THRESHOLD, 0.05))
        # - expected ...
        m.accept_order(Order(1, 0, "seller_c1_2", 1, 0.27, 0.04))

        print("\n")
        print(m.orders)
        assert m.get_grid_fee(bid_cluster=0, ask_cluster=1) == 0.003
        matches = m.match()
        print("\n")
        matches_df = pd.DataFrame(matches)
        print(matches_df.to_string())

        # plot_agg_flows(matches_df)

        '''
        Why does seller_c1_2 not sell to buyer_c1_3 ?
        - TM: buyer price < ask price. e.g. if buyer_c1_3 price was 0.04 it would match
          - TM: similarly, buyer_c0_1 would match if its price was 0.044
        
          type time     actor_id cluster                 energy price
        0   -1    0     buyer_MM    None  9223372036854775808.0  0.05
        1   -1    0   buyer_c0_1       0                    0.1  0.04
        2   -1    0   buyer_c1_3       1                   0.07  0.03
        3    1    0    seller_MM    None  9223372036854775808.0  0.05
        4    1    0  seller_c1_2       1                   0.27  0.04
        
        
           time  bid_id  ask_id bid_actor    ask_actor bid_cluster  ask_cluster  energy  price  included_grid_fee
        0     0       0       4  buyer_MM  seller_c1_2        None            1    0.27   0.05               0.01
        '''

        assert False


    def test_disputed_matching_approaches(self):
        # Highest price match is selected
        # ToDo not implemented
        # m = BestMarket(self.pn, time_step=0, disputed_matching='price')
        # # cluster 0
        # m.accept_order(Order(1, 0, 0, 0, 0.1, 1))
        # m.accept_order(Order(-1, 0, 4, 0, 0.1, 2))
        # # cluster 1
        # m.accept_order(Order(-1, 0, 3, 1, 0.1, 2))
        # matches = m.match()
        # assert matches[0]['included_grid_fee'] == 1

        # Match with the highest bid price is selected
        m = BestMarket(self.pn, time_step=0, disputed_matching='bid_price')
        # cluster 0
        m.accept_order(Order(1, 0, 0, 0, 0.1, 1))  # ask
        m.accept_order(Order(-1, 0, 4, 0, 0.1, 2))  # bid
        # cluster 1
        m.accept_order(Order(-1, 0, 3, 1, 0.1, 2))  # bid
        matches = m.match()
        assert matches[0]['included_grid_fee'] == 0

        # Disputed matches are resolved based on price
        m = BestMarket(self.pn, time_step=0, disputed_matching='grid_fee')
        # cluster 0
        m.accept_order(Order(1, 0, 0, 0, 0.1, 1))
        m.accept_order(Order(-1, 0, 4, 0, 0.1, 2))
        # cluster 1
        m.accept_order(Order(-1, 0, 3, 1, 0.1, 2))
        matches = m.match()
        assert matches[0]['included_grid_fee'] == 0

    def test_update_clusters(self):
        """Test the update of a cluster clearing price is correctly done when a better match with
        another cluster is found."""
        grid_fee_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        m = BestMarket(self.pn, grid_fee_matrix=grid_fee_matrix, time_step=0)

        # add bids for two clusters. cluster 2 has higher bids
        for price in range(20, 0, -1):
            m.accept_order(Order(-1, 0, price, 0, 0.1, price))
            m.accept_order(Order(-1, 0, price, 1, 0.1, price))

        # add asks. Both clusters have the same producers
        for price in range(8, 25, +1):
            m.accept_order(Order(1, 0, price, 0, 0.1, price))
            m.accept_order(Order(1, 0, price, 1, 0.1, price))

        matches = m.match()
        print([match["price"] for match in matches])
        print([match["bid_cluster"] for match in matches])
        print([match["ask_cluster"] for match in matches])
        assert all([match["price"] == matches[0]["price"] for match in matches])

    def test_profit_and_new_matching(self):
        """Test the update of a cluster clearing price is correctly done when a better match with
        another cluster is found."""
        # Scenario 0 / Simple case
        # bids and asks are in a single cluster. Get the amount of matches and clearing price
        grid_fee_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        m = BestMarket(self.pn, grid_fee_matrix=grid_fee_matrix, time_step=0)
        cfg.config.energy_unit = 0.01
        order_amount = 0.01
        for price in range(20, 0, -1):
            actor_id = price
            # "Order", ("type", "time", "actor_id", "cluster", "energy", "price"))
            m.accept_order(Order(-1, 0, actor_id, 0, order_amount, price))

        for price in range(0, 30, +1):
            price = max(price, 0.001)
            actor_id = price
            # "Order", ("type", "time", "actor_id", "cluster", "energy", "price"))
            m.accept_order(Order(1, 0, actor_id, 0, order_amount, price))

        matches = m.match()
        for match in matches:
            print(match)
        nr_matches_simple_case = len(matches)
        clearing_price_simple_case = matches[0]["price"]

        # Scenario 2
        # Simple case with two clusters with the same asks and bids as Scenario 0 in each cluster
        # Nr of matches should double and clearing price should be the same
        grid_fee_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        m = BestMarket(self.pn, grid_fee_matrix=grid_fee_matrix, time_step=0)
        order_amount = 0.01
        for price in range(20, 0, -1):
            actor_id = price
            # "Order", ("type", "time", "actor_id", "cluster", "energy", "price"))
            m.accept_order(Order(-1, 0, actor_id, 0, order_amount, price))
            m.accept_order(Order(-1, 0, actor_id, 1, order_amount, price))

        for price in range(0, 30, +1):
            price = max(price, 0.001)
            actor_id = price
            # "Order", ("type", "time", "actor_id", "cluster", "energy", "price"))
            m.accept_order(Order(1, 0, actor_id, 0, order_amount, price))
            m.accept_order(Order(1, 0, actor_id, 1, order_amount, price))

        matches = m.match()
        assert nr_matches_simple_case * 2 == len(matches)
        assert all([clearing_price_simple_case == match["price"] for match in matches])

        # Scenario 3
        # Simple case with two clusters with different bids. Since grid fee is 0, and same ask
        # prices exist in both clusters clearing price needs to be identical

        grid_fee_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        m = BestMarket(self.pn, grid_fee_matrix=grid_fee_matrix, time_step=0)
        order_amount = 0.01

        # BIDS
        for price in range(20, 0, -1):
            actor_id = price
            # "Order", ("type", "time", "actor_id", "cluster", "energy", "price"))
            m.accept_order(Order(-1, 0, actor_id * 2, 0, order_amount, price * 2))
            m.accept_order(Order(-1, 0, actor_id * 3, 1, order_amount, price * 3))

        # ASKS
        for price in range(0, 30, +1):
            price = max(price, 0.001)
            price = price
            actor_id = price
            # "Order", ("type", "time", "actor_id", "cluster", "energy", "price"))
            m.accept_order(Order(1, 0, actor_id, 0, order_amount, price))
            m.accept_order(Order(1, 0, actor_id, 1, order_amount, price))

        matches = m.match()
        for x in matches:
            print(x)

        assert all([abs(matches[0]["price"] - match["price"]) == 0 for match in matches])
        matched_energy_cluster_0 = sum([1 for match in matches if match["bid_cluster"] == 0])
        matched_energy_cluster_1 = sum([1 for match in matches if match["bid_cluster"] == 1])
        assert matched_energy_cluster_1 > matched_energy_cluster_0

        # Scenario 4
        # Simple case with two clusters with different bids. Since grid fee is 0, and same ask
        # prices exist in both clusters clearing price needs to be identical
        # Since bid prices of cluster 0 are lower than in scenario 3, the matched energy
        # in cluster 1 should increase.

        grid_fee_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        m = BestMarket(self.pn, grid_fee_matrix=grid_fee_matrix, time_step=0)
        order_amount = 0.01

        # BIDS
        for price in range(20, 0, -1):
            actor_id = price
            # "Order", ("type", "time", "actor_id", "cluster", "energy", "price"))
            m.accept_order(Order(-1, 0, actor_id * 2, 0, order_amount, price * 1))
            m.accept_order(Order(-1, 0, actor_id * 3, 1, order_amount, price * 3))

        # ASKS
        for price in range(0, 30, +1):
            price = max(price, 0.001)
            price = price
            actor_id = price
            # "Order", ("type", "time", "actor_id", "cluster", "energy", "price"))
            m.accept_order(Order(1, 0, actor_id, 0, order_amount, price))
            m.accept_order(Order(1, 0, actor_id, 1, order_amount, price))

        matches = m.match()
        assert all([abs(matches[0]["price"] - match["price"]) == 0 for match in matches])
        matched_energy_cluster_0 = sum([1 for match in matches if match["bid_cluster"] == 0])
        matched_energy_cluster_1_new = sum([1 for match in matches if match["bid_cluster"] == 1])
        assert matched_energy_cluster_1_new > matched_energy_cluster_0
        assert matched_energy_cluster_1_new > matched_energy_cluster_1

    # TODO revise
    '''
    def test_single_loop_multiple_bid_clusters(self):
        """Test the update of a cluster clearing price is correctly done when a better match with
        another cluster is found."""
        cfg.config.energy_unit = 0.01
        grid_fee2 = [[0, 0.01, 3],
                     [0.01, 0, 3],
                     [3, 3, 0]]

        fee = 0.01
        grid_fee1 = [[0, fee, fee],
                     [fee, 0, fee],
                     [fee, fee, 0]]

        grid_fee3 = [[0, 100, 100],
                     [100, 0, 100],
                     [100, 100, 0]]
        grid_fees = list([grid_fee1, grid_fee2, grid_fee3])
        mutation_nr = 1
        for g, grid_fee in enumerate(grid_fees):
            for bid_clusters in [1, 3]:
                for ask_clusters in [1, 3]:
                    for i in range(1, 3):
                        for per_cluster in [1, 3]:
                            mutation_nr = ((mutation_nr + 1) % 4) + 1
                            order_amount = 0.01 * (2 ** i)
                            m = self.create_market(order_amount=order_amount, grid_fee=grid_fee,
                                                   bid_clusters=bid_clusters,
                                                   ask_clusters=ask_clusters,
                                                   asks_per_cluster=per_cluster,
                                                   bids_per_cluster=per_cluster,
                                                   mutation_nr=mutation_nr)
                            matches = m.match()
                            self.print_summary(matches)
                            self.check_consistency(matches, grid_fee)
    '''

    def test_single_loop_multiple_ask_clusters(self):
        """Test the update of a cluster clearing price is correctly done when a better match with
        another cluster is found."""
        cfg.config.energy_unit = 0.01
        grid_fee2 = [[0, 0.01, 3],
                     [0.01, 0, 3],
                     [3, 3, 0]]

        grid_fee1 = [[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]]

        for grid_fee in [grid_fee1, grid_fee2]:
            for i in range(2, 3):
                order_amount = 0.01 * (2 ** i)
                m = self.create_market(order_amount=order_amount, grid_fee=grid_fee)
                matches2 = m.match()
                self.print_summary(matches2)

                self.print_matches(matches2)
                self.check_consistency(matches2, grid_fee)

    def print_matches(self, matches):
        clusters = {match["bid_cluster"] for match in matches}
        for cluster in clusters:
            cluster_matches = [m for m in matches if m["bid_cluster"] == cluster]
            cluster_matches = sorted(cluster_matches, key=lambda x: x["bid_actor"], reverse=True)
            if len(cluster_matches) == 0:
                print("no matches for cluster ", cluster)
                return
            print("Cluster ", cluster)
            print("Clearing Price ", cluster_matches[0]["price"])
            print("bid_actor", "ask_actor", "energy",
                  "ask_cluster", "included_grid_fee", "price", sep=",")
            for m in cluster_matches:
                print(m["bid_actor"],
                      m["ask_actor"],
                      m["energy"],
                      m["ask_cluster"],
                      m["included_grid_fee"],
                      m["price"], sep=",")

    def print_summary(self, matches):
        clusters = {match["bid_cluster"] for match in matches}
        for cluster in clusters:
            print(round(sum([m["energy"] for m in matches if m["bid_cluster"] == cluster]), 3),
                  end=" ")
        print({round(m["price"], 2) for m in matches})

    def compare_matches(self, m1, m2, show_matches=False):
        clusters = {match["bid_cluster"] for match in m1}
        print(str(["#"] * 40).replace("'", "").replace(",", ""))
        for match_type in [m1, m2]:
            self.print_summary(match_type)

        if show_matches:
            for match_type in [m1, m2]:
                self.print_matches(match_type)

        try:
            for cluster in clusters:
                matched_energy_m1 = sum([m["energy"] for m in m1 if m["bid_cluster"] == cluster])
                matched_energy_m2 = sum([m["energy"] for m in m2 if m["bid_cluster"] == cluster])
                assert matched_energy_m1 == pytest.approx(matched_energy_m2)

            for cluster in clusters:
                matched_energy_m1 = sum([m["energy"] for m in m1 if m["bid_cluster"] == cluster])
                matched_energy_m2 = sum([m["energy"] for m in m2 if m["bid_cluster"] == cluster])
                assert matched_energy_m1 == pytest.approx(matched_energy_m2)

                clear_prices_m1 = {m["price"] for m in m1 if m["bid_cluster"] == cluster}
                clear_prices_m2 = {m["price"] for m in m2 if m["bid_cluster"] == cluster}
                assert clear_prices_m1 == clear_prices_m2
                assert len(clear_prices_m1) == 1
        except AssertionError:
            for match_type in [m1, m2]:
                self.print_matches(match_type)

    def check_consistency(self, matches, grid_fees, show=False):
        clusters = {match["bid_cluster"] for match in matches}
        s_clusters = dict()
        # Note: Ask Actor id is equal to the actor price
        for cluster in clusters:
            s_clusters[cluster] = [match for match in matches if match["bid_cluster"] == cluster]
            s_clusters[cluster] = sorted(s_clusters[cluster], key=lambda x: x["ask_actor"])

        for m in matches:
            second_best_price = -float("inf")
            for cluster_ in clusters:
                try:
                    second_best_price = max(second_best_price,
                                            s_clusters[cluster_][-2]["ask_actor"] +
                                            s_clusters[cluster_][-2]["included_grid_fee"] -
                                            grid_fees[cluster_][m["ask_cluster"]])
                except IndexError:
                    pass
            assert m["price"] - m["included_grid_fee"] >= second_best_price
            if show:
                print(m["price"] - m["included_grid_fee"], " >= ", max(
                    [s_clusters[cluster_][-2]["ask_actor"] + s_clusters[cluster_][-2][
                        "included_grid_fee"] - grid_fees[cluster_][m["ask_cluster"]]
                     for cluster_ in clusters]))

        for m in matches:
            grid_fee = grid_fees[m["ask_cluster"]][m["bid_cluster"]]
            assert m["bid_actor"] >= m["price"], m
            assert m["ask_actor"] + grid_fee <= m["price"], m

    def create_market(self, order_amount=0.01, max_price=20, bid_clusters=1, ask_clusters=1,
                      bids_per_cluster=1, asks_per_cluster=1, mutation_nr=1, grid_fee=None):
        if grid_fee is None:
            grid_fee = [[0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 0]]
        grid_fee_matrix = [[v for v in fee] for fee in grid_fee]
        m = BestMarket(self.pn, grid_fee_matrix=grid_fee_matrix, time_step=0,
                       disputed_matching="grid_fee")
        order_amount = order_amount
        cfg.config.energy_unit = order_amount
        bids_mutator = [0, 0, -1.3, +5.5]
        asks_mutator = [0, +0.5, 1.1, +5.6]
        i = 0

        for price in range(max_price, 0, -1):
            for bid_cluster in range(bid_clusters):
                for nr_bid in range(bids_per_cluster):
                    current_price = price+bids_mutator[i % mutation_nr]
                    current_price = max(current_price, 0.001)
                    i += 1
                    m.accept_order(
                        Order(-1, 0, current_price, bid_cluster, order_amount, current_price))

        i = 0
        for price in range(0, max_price + 10, +1):
            for ask_cluster in range(ask_clusters):
                for nr_ask in range(asks_per_cluster):
                    current_price = price + asks_mutator[i % mutation_nr]
                    current_price = max(current_price, 0.001)
                    i += 1
                    # "Order", ("type", "time", "actor_id", "cluster", "energy", "price")
                    m.accept_order(
                        Order(+1, 0, current_price, ask_cluster, order_amount, current_price))
        return m
