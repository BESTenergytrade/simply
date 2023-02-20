from simply.actor import Order
from simply.market_2pac import TwoSidedPayAsClear

import pytest
import simply.config as cfg


class TestTwoSidedPayAsClear:
    cfg.Config("")

    def test_basic(self):
        """Tests the basic functionality of the TwoSidedPayAsClear object to accept bids and asks
        via the accept_order method and correctly match asks and bids when the match method
        is called."""
        m = TwoSidedPayAsClear(0, default_grid_fee=0)
        # no orders: no matches
        matches = m.match()
        assert len(matches) == 0

        # only one type: no match
        m.accept_order(Order(-1, 0, 0, None, 1, 1))
        matches = m.match()
        assert len(matches) == 0

        # bid and ask with same energy and price
        m.accept_order(Order(1, 0, 1, None, 1, 1))
        matches = m.match()
        assert len(matches) == 1
        # check match
        assert matches[0]["time"] == 0
        assert matches[0]["bid_actor"] == 0
        assert matches[0]["ask_actor"] == 1
        assert matches[0]["energy"] == 1
        assert matches[0]["price"] == 1

    def test_prices(self):
        """Tests that the highest bids are matched with the lowest asks and that all bids and
        asks above the crossover (when the bidding price becomes lower than the asking price)
        are matched on the clearing price."""
        # different prices
        m = TwoSidedPayAsClear(0, default_grid_fee=0)
        # ask above bid: no match
        m.accept_order(Order(-1, 0, 0, None, 1, 2))
        m.accept_order(Order(1, 0, 1, None, 1, 2.5))
        matches = m.match()
        assert len(matches) == 0

        # reset orders
        m.orders = m.orders[:0]
        # ask below bid: take lower one
        m.accept_order(Order(-1, 0, 0, None, 1, 2.5))
        m.accept_order(Order(1, 0, 1, None, 1, 2))
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["energy"] == 1
        assert matches[0]["price"] == 2

    def test_energy(self):
        """Tests that matches can be made when the amount of energy requested by the bid
        differs from the total amount of energy being offered by the ask."""
        # different energies
        m = TwoSidedPayAsClear(0, default_grid_fee=0)
        m.accept_order(Order(-1, 0, 0, None, .1, 1))
        m.accept_order(Order(1, 0, 1, None, 1, 1))
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["energy"] == 0.1

        m.orders = m.orders[:0]
        m.accept_order(Order(-1, 0, 0, None, 100, 1))
        m.accept_order(Order(1, 0, 1, None, .3, 1))
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["energy"] == pytest.approx(0.3)

        # reset orders
        m.orders = m.orders[:0]
        m.accept_order(Order(-1, 0, 1, None, 1, 7))
        m.accept_order(Order(-1, 0, 2, None, 1, 4))
        m.accept_order(Order(-1, 0, 3, None, 1, 3))
        m.accept_order(Order(1, 0, 4, None, 1, 1))
        m.accept_order(Order(1, 0, 5, None, 1, 1))
        m.accept_order(Order(1, 0, 6, None, 1, 5))

        matches = m.match()
        assert len(matches) == 2
        assert matches[0]["energy"] == 1
        assert matches[0]["price"] == 1

    def test_setting_order_id(self):
        # Check if matched orders retain original ID
        m = TwoSidedPayAsClear(0, default_grid_fee=0)
        m.accept_order(Order(-1, 0, 2, None, .2, 1), "ID1")
        m.accept_order(Order(1, 0, 3, None, 1, 1), "ID2")
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["energy"] == pytest.approx(0.2)
        assert matches[0]["bid_id"] == "ID1"
        assert matches[0]["ask_id"] == "ID2"

    def test_multiple(self):
        """Tests that multiple bids can be matched with one ask while there is available energy
        within the order."""
        # multiple bids to satisfy one ask
        m = TwoSidedPayAsClear(0, default_grid_fee=0)
        m.accept_order(Order(-1, 0, 1, None, 11, 1.1))
        m.accept_order(Order(-1, 0, 0, None, .1, 3))
        m.accept_order(Order(1, 0, 2, None, 2, 1))
        matches = m.match()
        assert len(matches) == 2
        assert matches[0]["energy"] == 0.1
        assert matches[1]["energy"] == 1.9  # only 2 in ask

        # multiple asks to satisfy one bid (order by price)
        m.orders = m.orders[:0]
        m.accept_order(Order(1, 0, 3, None, 30, 3))
        m.accept_order(Order(1, 0, 4, None, 50, 4))
        m.accept_order(Order(-1, 0, 0, None, 100, 5))
        m.accept_order(Order(1, 0, 1, None, 10, 1))
        m.accept_order(Order(1, 0, 2, None, 20, 2))

        matches = m.match()
        assert len(matches) == 4
        assert matches[0]["energy"] == 10
        assert matches[1]["energy"] == 20
        assert matches[2]["energy"] == 30
        assert matches[3]["energy"] == 40  # only 100 in bid

    def test_prices_matrix(self):
        # test prices with a given grid fee matrix
        # example: cost 1 for trade between clusters
        try:
            m = TwoSidedPayAsClear(0, grid_fee_matrix=[[0, 1], [1, 0]])
        except AssertionError:
            # assertion error because of a full grid_fee_matrix
            pass
        else:
            assert 0, "No assertion Error despite a grid_fee_matrix"

        # grid_fee matrix as a matrix
        m = TwoSidedPayAsClear(0, default_grid_fee=1)

        # grid-fees between nodes only allow for partial matching
        m.accept_order(Order(-1, 0, 2, 0, 1, 3))
        m.accept_order(Order(1, 0, 4, 1, 0.9, 3))
        m.accept_order(Order(1, 0, 0, 1, 0.1, 2))
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["energy"] == 0.1
        assert matches[0]["price"] == 3

        # grid_fee matrix as a value
        m = TwoSidedPayAsClear(0, default_grid_fee=0.5)

        # grid-fees between nodes only allow for partial matching
        m.accept_order(Order(-1, 0, 2, 0, 1, 3))
        m.accept_order(Order(1, 0, 4, 1, 0.9, 3))
        m.accept_order(Order(1, 0, 0, 1, 0.1, 2))
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["energy"] == 0.1
        assert matches[0]["price"] == 2.5
