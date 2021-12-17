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
        m = BestMarket(0, self.pn)
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
