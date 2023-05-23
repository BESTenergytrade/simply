from simply.actor import Order
from simply.market_fair import BestMarket, MARKET_MAKER_THRESHOLD, LARGE_ORDER_THRESHOLD
from simply.power_network import PowerNetwork
import networkx as nx
import pytest

from simply.scenario import Scenario


class TestBestMarket:
    nw = nx.Graph()
    nw.add_edges_from([(0, 1, {"weight": 1}), (1, 2), (1, 3), (0, 4)])
    pn = PowerNetwork("", nw, weight_factor=1)
    scenario = Scenario(None, [], None, [])

    def test_basic(self):
        """Tests the basic functionality of the BestMarket object to accept bids and asks via the
        accept_order method and correctly match asks and bids when the match method is called."""
        m = BestMarket(scenario=self.scenario, network=self.pn)
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
        m = BestMarket(scenario=self.scenario, network=self.pn)
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
        m = BestMarket(scenario=self.scenario, grid_fee_matrix=[[0, 1], [1, 0]])

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
        m = BestMarket(scenario=self.scenario, network=self.pn)
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
        m = BestMarket(scenario=self.scenario, network=self.pn)
        m.accept_order(Order(-1, 0, 2, None, .2, 1), "ID1")
        m.accept_order(Order(1, 0, 3, None, 1, 1), "ID2")
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["energy"] == pytest.approx(0.2)
        assert matches[0]["bid_id"] == "ID1"
        assert matches[0]["ask_id"] == "ID2"

    def test_setting_id_market_maker(self):
        # Check if matched orders retain original ID for selling or buying market makers
        m = BestMarket(scenario=self.scenario, network=self.pn)
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
        m = BestMarket(scenario=self.scenario, network=self.pn)
        m.accept_order(Order(-1, 0, 2, None, .1, 4))
        m.accept_order(Order(-1, 0, 3, None, 3, 3))
        m.accept_order(Order(1, 0, 4, None, 2, 1))
        matches = m.match()
        assert len(matches) == 2
        assert matches[0]["energy"] == pytest.approx(0.1)
        assert matches[1]["energy"] == pytest.approx(1.9)  # only 2 in ask

        # multiple asks to satisfy one bid
        m.orders = m.orders[:0]
        m.accept_order(Order(1, 0, 2, None, 10, 1))
        m.accept_order(Order(1, 0, 2, None, 20, 2))
        m.accept_order(Order(1, 0, 3, None, 30, 3))
        m.accept_order(Order(1, 0, 3, None, 50, 4))
        m.accept_order(Order(-1, 0, 4, None, 100, 5))
        matches = m.match()
        assert len(matches) == 4
        assert matches[0]["energy"] == pytest.approx(10)
        assert matches[1]["energy"] == pytest.approx(20)
        assert matches[2]["energy"] == pytest.approx(30)
        assert matches[3]["energy"] == pytest.approx(40)  # only 100 in bid

    def test_match_ordering(self):
        """Test to check that matching favors local orders in case of equal (adjusted) price."""
        m = BestMarket(scenario=self.scenario, network=self.pn)
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
        m = BestMarket(scenario=self.scenario, network=lpn)
        m.accept_order(Order(-1, 0, 6, None, 1, 5))
        m.accept_order(Order(1, 0, 3, None, 1, 4))
        m.accept_order(Order(1, 0, 4, None, 1, 3))
        matches = m.match()
        # match cluster must be closest to bid cluster
        assert matches[0]['bid_cluster'] == 2
        assert matches[0]['ask_cluster'] == 1

        # test that match doesn't prioritise local with price differential
        m = BestMarket(scenario=self.scenario, network=lpn)
        m.accept_order(Order(-1, 0, 6, None, 1, 100))
        m.accept_order(Order(1, 0, 3, None, 1, 50))
        m.accept_order(Order(1, 0, 4, None, 1, 3))
        matches = m.match()
        # match cluster must be closest to bid cluster
        assert matches[0]['price'] == 5

    def test_filter_large_orders(self):
        """Test to check that very large orders are ignored."""
        m = BestMarket(scenario=self.scenario, network=self.pn)
        m.accept_order(Order(-1, 0, 2, None, 1, 4))
        m.accept_order(Order(1, 0, 3, None, LARGE_ORDER_THRESHOLD + 1, 4))
        matches = m.match()
        # large ask is discarded, no match possible
        assert len(matches) == 0

    def test_market_maker_orders(self):
        """Test to check that market maker orders are not being ignored."""
        m = BestMarket(scenario=self.scenario, network=self.pn)
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
        m = BestMarket(scenario=self.scenario, network=self.pn)
        # add bids
        m.accept_order(Order(-1, 0, 1, 1, 0.1, 10))
        m.accept_order(Order(-1, 0, 1, 1, 0.1, 7))
        m.accept_order(Order(-1, 0, 0, 0, 0.1, 10))
        # add asks
        m.accept_order(Order(1, 0, 3, 1, 0.1, 6))
        m.accept_order(Order(1, 0, 3, 1, 0.1, 4))
        matches = m.match()
        assert len(matches) == 2
