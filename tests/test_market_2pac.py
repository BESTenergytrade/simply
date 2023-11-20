from simply.actor import Order
from simply.market_2pac import TwoSidedPayAsClear

import pytest

from simply.scenario import Scenario
from simply.market import MARKET_MAKER_THRESHOLD


class TestTwoSidedPayAsClear:
    scenario = Scenario(None, None, [])

    def test_basic(self):
        """Tests the basic functionality of the TwoSidedPayAsClear object to accept bids and asks
        via the accept_order method and correctly match asks and bids when the match method
        is called."""
        m = TwoSidedPayAsClear(time_step=0)
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
        m = TwoSidedPayAsClear(time_step=0)
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
        m = TwoSidedPayAsClear(time_step=0)
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
        m = TwoSidedPayAsClear(time_step=0, grid_fee_matrix=0)
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
        m = TwoSidedPayAsClear(time_step=0, grid_fee_matrix=0)
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

    def test_market_maker(self):
        # Test if inserting market_maker order works
        # example: cost 1 all trades
        m = TwoSidedPayAsClear(grid_fee_matrix=1, time_step=0)
        # Market Maker
        mm_price = 5
        m.accept_order(Order(-1, 0, "MarketMaker", 0, MARKET_MAKER_THRESHOLD, mm_price))
        m.accept_order(Order(1, 0, "MarketMaker", 1, MARKET_MAKER_THRESHOLD, mm_price))

        # grid-fees between nodes only allow for partial matching
        # Bids
        m.accept_order(Order(-1, 0, 1, 0, 5, mm_price * 3))
        m.accept_order(Order(-1, 0, 2, 0, 5, mm_price))

        # Asks
        m.accept_order(Order(1, 0, 3, 1, 3, mm_price * 2))
        m.accept_order(Order(1, 0, 4, 1, 2, mm_price))
        m.accept_order(Order(1, 0, 5, 1, 1, mm_price / 2))
        matches = m.match()
        # Bid actor 1 gets matched with 4,5 and the MM
        assert len(matches) == 3
        for m in matches:
            assert m["bid_actor"] == 1
            assert m["ask_actor"] in [4, 5, "MarketMaker"]
            assert m["price"] == mm_price + 1

        # Without a grid fee MarketMaker could match with itself
        # this should not happen
        m = TwoSidedPayAsClear(grid_fee_matrix=0, time_step=0)
        # Market Maker
        mm_price = 1
        m.accept_order(Order(-1, 0, "MarketMaker", 0, MARKET_MAKER_THRESHOLD, mm_price))
        m.accept_order(Order(1, 0, "MarketMaker", 1, MARKET_MAKER_THRESHOLD, mm_price))

        matches = m.match()
        # 1 bid gets matched, MM doesn't match with itself
        assert len(matches) == 0

        # but market maker should still match with other orders
        m = TwoSidedPayAsClear(grid_fee_matrix=0, time_step=0)
        # Market Maker
        mm_price = 1
        m.accept_order(Order(-1, 0, "MarketMaker", 0, MARKET_MAKER_THRESHOLD, mm_price))
        m.accept_order(Order(1, 0, "MarketMaker", 1, MARKET_MAKER_THRESHOLD, mm_price))

        # grid-fees between nodes only allow for partial matching
        # Bids
        m.accept_order(Order(-1, 0, 2, 0, 3, mm_price * 3))
        matches = m.match()
        # 1 bid gets matched, MM doesn't match with itself
        assert len(matches) == 1

    def test_prices_matrix(self):
        # test prices with a given grid fee matrix
        # "Assertion Error because grid_fee_matrix"
        with pytest.raises(AssertionError, ):
            m = TwoSidedPayAsClear(0, grid_fee_matrix=[[0, 1], [1, 0]])

        # example: cost 1 for trade between clusters
        m = TwoSidedPayAsClear(grid_fee_matrix=1, time_step=0)

        # grid-fees only allow for partial matching
        m.accept_order(Order(-1, 0, 2, 0, 1, 3))
        m.accept_order(Order(1, 0, 4, 1, 0.9, 3))
        m.accept_order(Order(1, 0, 0, 1, 0.1, 2))
        matches = m.match()
        assert len(matches) == 1
        assert matches[0]["energy"] == 0.1
        assert matches[0]["price"] == 3
        assert matches[0]["included_grid_fee"] == 1

        # default grid fee should only be applied once
        # example: cost 1 for trade between clusters
        m = TwoSidedPayAsClear(grid_fee_matrix=1, time_step=0)

        # grid-fees only allow for partial matching
        # type, time, actor_id, cluster, amount, price
        m.accept_order(Order(-1, 0, 2, 0, 1.5, 3.8))
        m.accept_order(Order(-1, 0, 3, 0, 1, 3))
        m.accept_order(Order(-1, 0, 3, 0, 1, 2))
        m.accept_order(Order(1, 0, 0, 1, 2, 2))
        matches = m.match()
        assert len(matches) == 2
        assert matches[0]["energy"] == 1.5
        assert matches[1]["energy"] == 0.5
        assert matches[0]["price"] == 3
