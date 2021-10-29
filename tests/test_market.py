from simply.actor import Order
from simply.market import Market

import pytest

class TestMarket:

    def test_init(self):
        m = Market(0)
        assert m.t == 0

    def test_accept_order(self):
        # Order(type, time, actor_id, energy, price), callback
        m= Market(0)
        m.accept_order(Order(1,0,0,1,1), None)
        m.accept_order(Order(-1,0,0,1,1), sum)
        # reject orders from the future
        with pytest.raises(Exception) as e:
            m.accept_order(Order(1,1,0,1,1), None)
        # reject orders from the past
        with pytest.raises(Exception) as e:
            m.accept_order(Order(-1,1,0,1,1), None)
        # reject wrong Order type
        with pytest.raises(ValueError) as e:
            m.accept_order(Order(0,0,0,1,1), None)

    def test_get_orders(self):
        m = Market(0)
        # no entries
        assert m.get_order_df().shape[0] == 0
        m.accept_order(Order(1,0,0,1,1), None)
        # one row
        assert m.get_order_df().shape[0] == 1
        # order has five elements
        assert m.get_order_df().shape[1] == 5
        m.accept_order(Order(-1,0,0,1,1), None)
        # two entries
        assert m.get_order_df().shape[0] == 2

    def test_get_bids(self):
        m = Market(0)
        assert m.get_bids().shape[0] == 0
        # add ask
        m.accept_order(Order(-1,0,0,1,1), None)
        assert m.get_bids().shape[0] == 0
        # add bid
        m.accept_order(Order(1,0,0,1,1), None)
        assert m.get_bids().shape[0] == 1
        # and one more bid
        m.accept_order(Order(1,0,1,2,1), None)
        assert m.get_bids().shape[0] == 2

    def test_get_asks(self):
        m = Market(0)
        assert m.get_asks().shape[0] == 0
        # add bid
        m.accept_order(Order(1,0,0,1,1), None)
        assert m.get_asks().shape[0] == 0
        # add ask
        m.accept_order(Order(-1,0,0,1,1), None)
        assert m.get_asks().shape[0] == 1
        # and one more bid
        m.accept_order(Order(-1,0,1,2,1), None)
        assert m.get_asks().shape[0] == 2

    def test_match(self):
        m = Market(0)
        assert len(m.match()) == 0
        # bid and ask with same energy and price
        m.accept_order(Order(1,0,0,1,1), None)
        m.accept_order(Order(-1,0,1,1,1), None)
        matches = m.match()
        assert len(matches) == 1
        # check energy and price
        assert True

        # check different prices, different energies, multiple bids to satisfy one ask, ...

    def test_clear(self):
        pass
