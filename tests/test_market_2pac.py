from simply.actor import Order
from simply.market_2pac import TwoSidedPayAsClear

import pytest


class TestTwoSidedPayAsClear:

    def test_basic(self):
        """Tests the basic functionality of the TwoSidedPayAsClear object to accept bids and asks via the accept_order
            method and correctly match asks and bids when the match method is called."""
        m = TwoSidedPayAsClear(0)
        # no orders: no matches
        matches = m.match()
        assert len(matches) == 0

        # only one type: no match
        m.accept_order(Order(-1,0,0,1,1), None)
        matches = m.match()
        assert len(matches) == 0

        # bid and ask with same energy and price
        m.accept_order(Order(1,0,1,1,1), None)
        matches = m.match()
        assert len(matches) == 1
        # check match
        assert matches[0]["time"] == 0
        assert matches[0]["bid_actor"] == 0
        assert matches[0]["ask_actor"] == 1
        assert matches[0]["energy"] == 1
        assert matches[0]["price"] == 1

    def test_prices(self):
        """Tests that the highest bids are matched with the lowest asks and that all bids and asks above the crossover
        (when the bidding price becomes lower than the asking price) are matched on the clearing price."""
        # different prices
        m = TwoSidedPayAsClear(0)
        # ask above bid: no match
        m.accept_order(Order(-1,0,0,1,2), None)
        m.accept_order(Order(1,0,1,1,2.5), None)
        matches = m.match()
        assert len(matches) == 0

        # reset orders
        m.orders = m.orders[:0]
        # ask below bid: take lower one
        m.accept_order(Order(-1,0,0,1,2.5), None)
        m.accept_order(Order(1,0,1,1,2), None)
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["energy"] == 1
        assert matches[0]["price"] == 2

        # reset orders
        m.orders = m.orders[:0]
        m.accept_order(Order(-1, 0, 1, 1, 7), None)
        m.accept_order(Order(-1, 0, 2, 1, 4), None)
        m.accept_order(Order(-1, 0, 3, 1, 3), None)
        m.accept_order(Order(1, 0, 4, 1, 1), None)
        m.accept_order(Order(1, 0, 5, 1, 1), None)
        m.accept_order(Order(1, 0, 6, 1, 5), None)

        matches = m.match()
        assert len(matches) == 2
        assert matches[0]["energy"] == 1
        assert matches[0]["price"] == 1

    def test_energy(self):
        """Tests that matches can be made when the amount of energy requested by the bid differs from the total
             amount of energy being offered by the ask."""
        # different energies
        m = TwoSidedPayAsClear(0)
        m.accept_order(Order(-1,0,0,.1,1), None)
        m.accept_order(Order(1,0,1,1,1), None)
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["energy"] == 0.1

        m.orders = m.orders[:0]
        m.accept_order(Order(-1,0,0,100,1), None)
        m.accept_order(Order(1,0,1,.3,1), None)
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["energy"] == pytest.approx(0.3)

    def test_multiple(self):
        """Tests that multiple bids can be matched with one ask while there is available energy within the order."""
        # multiple bids to satisfy one ask
        m = TwoSidedPayAsClear(0)
        m.accept_order(Order(-1, 0, 1, 11, 1.1), None)
        m.accept_order(Order(-1,0,0,.1,3), None)
        m.accept_order(Order(1,0,2,2,1), None)
        matches = m.match()
        assert len(matches) == 2
        assert matches[0]["energy"] == 0.1
        assert matches[1]["energy"] == 1.9  # only 2 in ask

        # multiple asks to satisfy one bid (order by price)
        m.orders = m.orders[:0]
        m.accept_order(Order(1, 0, 3, 30, 3), None)
        m.accept_order(Order(1, 0, 4, 50, 4), None)
        m.accept_order(Order(-1, 0, 0, 100, 5), None)
        m.accept_order(Order(1,0,1,10,1), None)
        m.accept_order(Order(1,0,2,20,2), None)

        matches = m.match()
        assert len(matches) == 4
        assert matches[0]["energy"] == 10
        assert matches[1]["energy"] == 20
        assert matches[2]["energy"] == 30
        assert matches[3]["energy"] == 40  # only 100 in bid
