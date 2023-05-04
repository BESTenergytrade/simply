from simply.actor import Order
from simply.market_fair import BestMarket, MARKET_MAKER_THRESHOLD, LARGE_ORDER_THRESHOLD
from simply.power_network import PowerNetwork
import networkx as nx
import pytest
from simply.config import Config


class TestBestMarket:
    nw = nx.Graph()
    nw.add_edges_from([(0, 1, {"weight": 1}), (1, 2), (1, 3), (0, 4)])
    pn = PowerNetwork("", nw, weight_factor=1)
    Config("")

    def test_basic(self):
        """Tests the basic functionality of the BestMarket object to accept bids and asks via the
        accept_order method and correctly match asks and bids when the match method is called."""
        m = BestMarket(0, self.pn)
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
        m = BestMarket(0, self.pn)
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
        m = BestMarket(0, grid_fee_matrix=[[0, 1], [1, 0]])

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
        m = BestMarket(0, self.pn)
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
        m = BestMarket(0, self.pn)
        m.accept_order(Order(-1, 0, 2, None, .2, 1), "ID1")
        m.accept_order(Order(1, 0, 3, None, 1, 1), "ID2")
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["energy"] == pytest.approx(0.2)
        assert matches[0]["bid_id"] == "ID1"
        assert matches[0]["ask_id"] == "ID2"

    def test_setting_id_market_maker(self):
        # Check if matched orders retain original ID for selling or buying market makers
        m = BestMarket(0, self.pn)
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
        m = BestMarket(0, self.pn)
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
        m = BestMarket(0, self.pn)
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
        m = BestMarket(0, lpn)
        m.accept_order(Order(-1, 0, 6, None, 1, 5))
        m.accept_order(Order(1, 0, 3, None, 1, 4))
        m.accept_order(Order(1, 0, 4, None, 1, 3))
        matches = m.match()
        # match cluster must be closest to bid cluster
        assert matches[0]['bid_cluster'] == 2
        assert matches[0]['ask_cluster'] == 1

        # test that match doesn't prioritise local with price differential
        m = BestMarket(0, lpn)
        m.accept_order(Order(-1, 0, 6, None, 1, 100))
        m.accept_order(Order(1, 0, 3, None, 1, 50))
        m.accept_order(Order(1, 0, 4, None, 1, 3))
        matches = m.match()
        # match cluster must be closest to bid cluster
        assert matches[0]['price'] == 5

    def test_filter_large_orders(self):
        """Test to check that very large orders are ignored."""
        m = BestMarket(0, self.pn)
        m.accept_order(Order(-1, 0, 2, None, 1, 4))
        m.accept_order(Order(1, 0, 3, None, LARGE_ORDER_THRESHOLD + 1, 4))
        matches = m.match()
        # large ask is discarded, no match possible
        assert len(matches) == 0

    def test_market_maker_orders(self):
        """Test to check that market maker orders are not being ignored."""
        m = BestMarket(0, self.pn)
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
        m = BestMarket(0, self.pn)
        # add bids
        m.accept_order(Order(-1, 0, 0, 1, 0.1, 10))
        m.accept_order(Order(-1, 0, 1, 1, 0.1, 7))
        m.accept_order(Order(-1, 0, 0, 0, 0.1, 10))
        # add asks
        m.accept_order(Order(1, 0, 3, 1, 0.1, 6))
        m.accept_order(Order(1, 0, 3, 1, 0.1, 4))
        matches = m.match()
        assert all([match["bid_actor"] == 0 for match in matches])
        matched_energy = sum([match["energy"] for match in matches])
        assert matched_energy == pytest.approx(0.2)

    def test_disputed_matching_approaches(self):
        # Highest price match is selected
        m = BestMarket(0, self.pn, disputed_matching='price')
        # cluster 0
        m.accept_order(Order(1, 0, 0, 0, 0.1, 1))
        m.accept_order(Order(-1, 0, 4, 0, 0.1, 2))
        # cluster 1
        m.accept_order(Order(-1, 0, 3, 1, 0.1, 2))
        matches = m.match()
        assert matches[0]['included_grid_fee'] == 1

        # Match with the highest bid price is selected
        m = BestMarket(0, self.pn, disputed_matching='bid_price')
        # cluster 0
        m.accept_order(Order(1, 0, 0, 0, 0.1, 1))
        m.accept_order(Order(-1, 0, 4, 0, 0.1, 2))
        # cluster 1
        m.accept_order(Order(-1, 0, 3, 1, 0.1, 2))
        matches = m.match()
        assert matches[0]['included_grid_fee'] == 0

        # Disputed matches are resolved based on price
        m = BestMarket(0, self.pn, disputed_matching='grid_fee')
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
        grid_fee_matrix = [[0, 0.0000, 0], [0.0000, 0, 0], [0, 0, 0]]
        m = BestMarket(0, self.pn, grid_fee_matrix=grid_fee_matrix)

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
        m = BestMarket(0, self.pn, grid_fee_matrix=grid_fee_matrix, disputed_matching='profit')
        order_amount = 0.01
        for price in range(20, 0, -1):
            actor_id = price
            # "Order", ("type", "time", "actor_id", "cluster", "energy", "price"))
            m.accept_order(Order(-1, 0, actor_id, 0, order_amount, price))

        for price in range(0, 30, +1):
            price = max(price, 0.001)
            price = price
            actor_id = price
            #"Order", ("type", "time", "actor_id", "cluster", "energy", "price"))
            m.accept_order(Order(1, 0, actor_id, 0, order_amount, price))

        matches = m.match()
        nr_matches_simple_case = len(matches)
        clearing_price_simple_case = matches[0]["price"]

        # Scenario 2
        # Simple case with two clusters with the same asks and bids as Scenario 0 in each cluster
        # Nr of matches should double and clearing price should be the same
        grid_fee_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        m = BestMarket(0, self.pn, grid_fee_matrix=grid_fee_matrix, disputed_matching='profit')
        order_amount = 0.01
        for price in range(20, 0, -1):
            actor_id = price
            # "Order", ("type", "time", "actor_id", "cluster", "energy", "price"))
            m.accept_order(Order(-1, 0, actor_id, 0, order_amount, price))
            m.accept_order(Order(-1, 0, actor_id, 1, order_amount, price))

        for price in range(0, 30, +1):
            price = max(price, 0.001)
            price = price
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
        m = BestMarket(0, self.pn, grid_fee_matrix=grid_fee_matrix, disputed_matching='profit')
        order_amount = 0.01

        # BIDS
        for price in range(20, 0, -1):
            actor_id = price
            # "Order", ("type", "time", "actor_id", "cluster", "energy", "price"))
            m.accept_order(Order(-1, 0, actor_id*2, 0, order_amount, price*2))
            m.accept_order(Order(-1, 0, actor_id*3, 1, order_amount, price*3))

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

        assert all([abs(matches[0]["price"]-match["price"]) == 0  for match in matches])
        matched_energy_cluster_0 = sum([1 for match in matches if match["bid_cluster"] == 0])
        matched_energy_cluster_1 = sum([1 for match in matches if match["bid_cluster"] == 1])
        assert matched_energy_cluster_1 > matched_energy_cluster_0

        # Scenario 4
        # Simple case with two clusters with different bids. Since grid fee is 0, and same ask
        # prices exist in both clusters clearing price needs to be identical
        # Since bid prices of cluster 0 are lower than in scenario 3, the matched energy
        # in cluster 1 should increase.

        grid_fee_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        m = BestMarket(0, self.pn, grid_fee_matrix=grid_fee_matrix, disputed_matching='profit')
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
        for x in matches:
            print(x)

        assert all([abs(matches[0]["price"] - match["price"]) == 0 for match in matches])
        matched_energy_cluster_0 = sum([1 for match in matches if match["bid_cluster"] == 0])
        matched_energy_cluster_1_new = sum([1 for match in matches if match["bid_cluster"] == 1])
        assert matched_energy_cluster_1_new > matched_energy_cluster_0
        assert matched_energy_cluster_1_new > matched_energy_cluster_1

    def test_scenario_5(self):
        # Scenario 5
        # Simple case with two clusters with different bids. Since grid fee is 0, and same ask
        # prices exist in both clusters clearing price needs to be identical


        grid_fee_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
        m = BestMarket(0, self.pn, grid_fee_matrix=grid_fee_matrix, disputed_matching='profit')
        order_amount = 0.03

        # BIDS
        for price in range(20, 0, -1):
            price1 = price*2
            actor_id = price1
            # "Order", ("type", "time", "actor_id", "cluster", "energy", "price"))
            m.accept_order(Order(-1, 0, actor_id, 0, order_amount, price1))
            price2 = price*3
            actor_id = price2
            m.accept_order(Order(-1, 0, actor_id, 1, order_amount, price2))

        # ASKS
        for price in range(0, 30, +1):
            price = max(price, 0.001)
            # "Order", ("type", "time", "actor_id", "cluster", "energy", "price"))
            price1 = price+1/2
            actor_id = price1
            m.accept_order(Order(1, 0, actor_id, 0, order_amount, price1))
            price2 = price
            actor_id = price2
            m.accept_order(Order(1, 0, actor_id, 1, order_amount, price2))

        matches = m.match()
        for x in matches:
            print(x)

        # assert all([abs(matches[0]["price"] - match["price"]) == 0 for match in matches])
        matched_energy_cluster_0 = sum([1 for match in matches if match["bid_cluster"] == 0])
        matched_energy_cluster_1 = sum([1 for match in matches if match["bid_cluster"] == 1])
#
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)
# t=TestBestMarket().test_scenario_5()