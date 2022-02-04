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

    def test_not_accept_existing_order_id(self):
        # The order ID is used twice, but should be unique -> else raise ValueError
        m = BestMarket(0, self.pn)
        m.accept_order(Order(-1, 0, 2, .2, 1), None, "ID1")
        with pytest.raises(ValueError):
            m.accept_order(Order(1, 0, 3, 1, 1), None, "ID1")

    def test_setting_order_id(self):
        # Check if matched orders retain original ID
        m = BestMarket(0, self.pn)
        m.accept_order(Order(-1,0,2,.2,1), None, "ID1")
        m.accept_order(Order(1,0,3,1,1), None, "ID2")
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["energy"] == pytest.approx(0.2)
        assert matches[0]["bid_id"] == "ID1"
        assert matches[0]["ask_id"] == "ID2"

    def test_setting_id_market_maker(self):
        # Check if matched orders retain original ID for selling or buying market makers
        m = BestMarket(0, self.pn)
        # Test asking market maker with order ID
        m.accept_order(Order(-1,0,2,.3,1), None, "ID1")
        m.accept_order(Order(1,0,3,9223372036854775807,1), None, "ID2")
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["energy"] == pytest.approx(0.3)
        assert matches[0]["bid_id"] == "ID1"
        assert matches[0]["ask_id"] == "ID2"

        # Reset orders
        m.orders = m.orders[:0]
        # Test bidding market maker with order ID
        m.accept_order(Order(-1, 0, 2, 9223372036854775807, 1), None, "ID3")
        m.accept_order(Order(1, 0, 3, .3, 1), None, "ID4")
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["energy"] == pytest.approx(0.3)
        assert matches[0]["bid_id"] == "ID3"
        assert matches[0]["ask_id"] == "ID4"

    def test_multiple(self):
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
