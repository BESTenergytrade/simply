from simply.actor import Order
from simply.market_fair import BestMarket
from simply.power_network import PowerNetwork
import networkx as nx
import pytest


class TestBestMarket:

    nw = nx.Graph()
    nw.add_edges_from([(0,1, {"weight": 1}), (1,2), (1,3), (0,4)])
    pn = PowerNetwork("", nw)

    def test_basic(self):
        """Tests the basic functionality of the BestMarket object to accept bids and asks via the accept_order method
            and correctly match asks and bids when the match method is called."""
        m = BestMarket(0, self.pn)
        # no orders: no matches
        matches = m.match()
        assert len(matches) == 0

        # only one type: no match
        m.accept_order(Order(-1,0,2,1,1), None)
        matches = m.match()
        assert len(matches) == 0

        # bid and ask with same energy and price
        m.accept_order(Order(1,0,3,1,1), None)
        matches = m.match()
        assert len(matches) == 1
        # check match
        assert matches[0]["time"] == 0
        assert matches[0]["bid_actor"] == 2
        assert matches[0]["ask_actor"] == 3
        assert matches[0]["energy"] == pytest.approx(1)
        assert matches[0]["price"] == pytest.approx(1)

    def test_prices(self):
        """Tests that the prices of the orders are correctly affected by the weights of the PowerNetwork."""
        # different prices
        m = BestMarket(0, self.pn, 1)
        # ask above bid: no match
        m.accept_order(Order(-1,0,2,1,2), None)
        m.accept_order(Order(1,0,3,1,2.5), None)
        matches = m.match()
        assert len(matches) == 0

        # reset orders
        m.orders = m.orders[:0]
        # ask below bid: take highest asking price
        m.accept_order(Order(-1,0,2,1,2.5), None)
        m.accept_order(Order(1,0,3,1,2), None)
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["energy"] == pytest.approx(1)
        assert matches[0]["price"] == pytest.approx(2)

        m.orders = m.orders[:0]
        # weight between nodes too high
        m.accept_order(Order(-1,0,2,1,3), None)
        m.accept_order(Order(1,0,4,1,3), None)
        matches = m.match()
        assert len(matches) == 0

        m.orders = m.orders[:0]
        # weight between nodes low enough
        m.accept_order(Order(-1,0,4,1,3), None)
        m.accept_order(Order(1,0,2,1,2), None)
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["energy"] == pytest.approx(1)
        assert matches[0]["price"] == pytest.approx(3)  # 2 + weight(1)

        m.orders = m.orders[:0]
        # match different clusters, even though there are orders from same cluster
        m.accept_order(Order(1,0,2,1,2), None)
        m.accept_order(Order(-1,0,3,1,2), None)
        m.accept_order(Order(-1,0,4,1,4), None)
        # expected: match 2 and 4, even though 2 and 3 are in same cluster (worse conditions)
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["ask_actor"] == 2
        assert matches[0]["bid_actor"] == 4
        assert matches[0]["energy"] == pytest.approx(1)
        assert matches[0]["price"] == pytest.approx(3)  # 2 + weight(1)

    def test_undefined(self):
        """Tests correct accounting for the sum of energy and price."""
        # same price: depends on internal ordering
        m = BestMarket(0, self.pn, 1)
        m.accept_order(Order(-1,0,2,1,5), None)
        m.accept_order(Order(1,0,3,1,4), None)
        m.accept_order(Order(1,0,4,1,3), None)
        matches = m.match()
        # sum of energies must match, price must be identical
        assert sum([m["energy"] for m in matches]) == pytest.approx(1)
        for m in matches:
            assert m["price"] == pytest.approx(4)

    def test_energy(self):
        """Tests that the amount of energy traded equals the maximum amount available that is less than or equal to
            the amount requested by the bid."""
        # different energies
        m = BestMarket(0, self.pn)
        m.accept_order(Order(-1,0,2,.1,1), None)
        m.accept_order(Order(1,0,3,1,1), None)
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["energy"] == pytest.approx(0.1)

        m.orders = m.orders[:0]
        m.accept_order(Order(-1,0,2,100,1), None)
        m.accept_order(Order(1,0,3,.3,1), None)
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["energy"] == pytest.approx(0.3)

    def test_multiple(self):
        """Tests that matches can be made which require multiple asks to satisfy one bid or multiple bids to
            satisfy one ask."""
        # multiple bids to satisfy one ask
        m = BestMarket(0, self.pn)
        m.accept_order(Order(-1,0,2,.1,4), None)
        m.accept_order(Order(-1,0,3,3,3), None)
        m.accept_order(Order(1,0,4,2,1), None)
        matches = m.match()
        assert len(matches) == 2
        assert matches[0]["energy"] == pytest.approx(0.1)
        assert matches[1]["energy"] == pytest.approx(1.9)  # only 2 in ask

        # multiple asks to satisfy one bid
        m.orders = m.orders[:0]
        m.accept_order(Order(1,0,2,10,1), None)
        m.accept_order(Order(1,0,2,20,2), None)
        m.accept_order(Order(1,0,3,30,3), None)
        m.accept_order(Order(1,0,3,50,4), None)
        m.accept_order(Order(-1,0,4,100,5), None)
        matches = m.match()
        assert len(matches) == 4
        assert matches[0]["energy"] == pytest.approx(10)
        assert matches[1]["energy"] == pytest.approx(20)
        assert matches[2]["energy"] == pytest.approx(30)
        assert matches[3]["energy"] == pytest.approx(40)  # only 100 in bid

    def test_match_ordering(self):
        """Test to check that matching favors local orders in case of equal (adjusted) price."""
        m = BestMarket(0, self.pn, 1)
        m.accept_order(Order(-1, 0, 2, 1, 4), None)
        m.accept_order(Order(1, 0, 3, 1, 4), None)
        m.accept_order(Order(1, 0, 4, 1, 3), None)
        matches = m.match()
        # match cluster must be closest to bid cluster
        assert m.node_to_cluster[matches[0]['ask_actor']] == 1

        # test across multiple clusters
        lnw = nx.Graph()
        lnw.add_edges_from([(0, 1, {"weight": 1}), (1, 2), (1, 3), (0, 4), (2, 5, {"weight": 1}), (5, 6)])
        lpn = PowerNetwork("", lnw)
        m = BestMarket(0, lpn, 1)
        m.accept_order(Order(-1, 0, 6, 1, 5), None)
        m.accept_order(Order(1, 0, 3, 1, 4), None)
        m.accept_order(Order(1, 0, 4, 1, 3), None)
        matches = m.match()
        # match cluster must be closest to bid cluster
        assert m.node_to_cluster[matches[0]['ask_actor']] == 1

        # test that match doesn't prioritise local with price differential
        m = BestMarket(0, lpn, 1)
        m.accept_order(Order(-1, 0, 6, 1, 100), None)
        m.accept_order(Order(1, 0, 3, 1, 50), None)
        m.accept_order(Order(1, 0, 4, 1, 3), None)
        matches = m.match()
        # match cluster must be closest to bid cluster
        assert m.node_to_cluster[matches[0]['ask_actor']] == 0

    def test_filter_large_orders(self):
        """Test to check that very large orders are ignored."""
        m = BestMarket(0, self.pn, 1)
        large_order_threshold = 2**32
        m.accept_order(Order(-1, 0, 2, 1, 4), None)
        m.accept_order(Order(1, 0, 3, large_order_threshold+1, 4), None)
        matches = m.match()
        # large ask is discarded, no match possible
        assert len(matches) == 0

    def test_market_maker_orders(self):
        """Test to check that market maker orders are not being ignored."""
        m = BestMarket(0, self.pn, 1)
        market_maker_threshold = 2**63-1
        m.accept_order(Order(-1, 0, 2, 1, 4), None)
        m.accept_order(Order(1, 0, 3, market_maker_threshold+1, 4), None)
        matches = m.match()
        # matched with market maker
        assert len(matches) == 1
        assert matches[0]['energy'] == pytest.approx(1)
        assert matches[0]['price'] == pytest.approx(4)
