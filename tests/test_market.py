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
        assert len(m.orders) == 0
        m.accept_order(Order(1,0,0,1,1), None)
        assert len(m.orders) == 1
        m.accept_order(Order(-1,0,0,1,1), sum)
        assert len(m.orders) == 2
        # reject orders from the future
        with pytest.raises(ValueError):
            m.accept_order(Order(1,1,0,1,1), None)
        # reject orders from the past
        with pytest.raises(ValueError):
            m.accept_order(Order(-1,1,0,1,1), None)
        # reject wrong Order type
        with pytest.raises(ValueError):
            m.accept_order(Order(0,0,0,1,1), None)

    def test_get_bids(self):
        m = Market(0)
        assert m.get_bids().shape[0] == 0
        # add ask
        m.accept_order(Order(1,0,0,1,1), None)
        assert m.get_bids().shape[0] == 0
        # add bid
        m.accept_order(Order(-1,0,0,1,1), None)
        assert m.get_bids().shape[0] == 1
        # and one more bid
        m.accept_order(Order(-1,0,1,2,1), None)
        assert m.get_bids().shape[0] == 2

    def test_get_asks(self):
        m = Market(0)
        assert m.get_asks().shape[0] == 0
        # add bid
        m.accept_order(Order(-1,0,0,1,1), None)
        assert m.get_asks().shape[0] == 0
        # add ask
        m.accept_order(Order(1,0,0,1,1), None)
        assert m.get_asks().shape[0] == 1
        # and one more bid
        m.accept_order(Order(1,0,1,2,1), None)
        assert m.get_asks().shape[0] == 2

    def test_clear(self):
        m = Market(0)
        m.accept_order(Order(-1,0,0,1,1), None)
        # no match possible (only one order)
        m.clear(reset=False)
        # new list of matches saved
        assert len(m.matches) == 1
        # no matches in list
        assert len(m.matches[0]) == 0
        # order must still be in market
        assert m.orders.shape[0] == 1
        m.clear(reset=True)
        # another list of matches saved
        assert len(m.matches) == 2
        # still no match
        assert len(m.matches[1]) == 0
        # orders are reset
        assert m.orders.shape[0] == 0


class TestPayAsBid():

    def test_basic(self):
        m = Market(0)
        # no orders
        m.define_order_id()
        assert len(m.match(m.get_order_dict())) == 0
        # bid and ask with same energy and price
        m.accept_order(Order(-1,0,0,1,1), None)
        m.accept_order(Order(1,0,1,1,1), None)
        m.define_order_id()
        matches = m.match(m.get_order_dict())
        assert len(matches) == 1
        # check match
        assert matches[0]["time"] == 0
        assert matches[0]["bid_actor"] == 0
        assert matches[0]["ask_actor"] == 1
        assert matches[0]["energy"] == 1
        assert matches[0]["price"] == 1

    def test_prices(self):
        # different prices, pay as bid
        m = Market(0)
        # ask above bid: no match
        m.accept_order(Order(-1,0,0,1,2), None)
        m.accept_order(Order(1,0,1,1,2.5), None)
        m.define_order_id()
        matches = m.match(m.get_order_dict())
        assert len(matches) == 0

        # reset orders
        m.orders = m.orders[:0]
        # ask below bid: match with bid price
        m.accept_order(Order(-1,0,0,1,2), None)
        m.accept_order(Order(1,0,1,1,.5), None)
        m.define_order_id()
        matches = m.match(m.get_order_dict())
        assert len(matches) == 1
        assert matches[0]["energy"] == 1
        assert matches[0]["price"] == 2

    def test_energy(self):
        # different energies
        m = Market(0)
        m.accept_order(Order(-1,0,0,.1,1), None)
        m.accept_order(Order(1,0,1,1,1), None)
        m.define_order_id()
        matches = m.match(m.get_order_dict())
        assert len(matches) == 1
        assert matches[0]["energy"] == 0.1

        m.orders = m.orders[:0]
        m.accept_order(Order(-1,0,0,100,1), None)
        m.accept_order(Order(1,0,1,.3,1), None)
        m.define_order_id()
        matches = m.match(m.get_order_dict())
        assert len(matches) == 1
        assert matches[0]["energy"] == 0.3

    def test_multiple(self):
        # multiple bids to satisfy one ask
        m = Market(0)
        m.accept_order(Order(-1,0,0,.1,1), None)
        m.accept_order(Order(-1,0,1,11,1), None)
        m.accept_order(Order(1,0,2,2,1), None)
        m.define_order_id()
        matches = m.match(m.get_order_dict())
        assert len(matches) == 2
        assert matches[0]["energy"] == 0.1
        assert matches[1]["energy"] == 1.9  # only 2 in ask

        # multiple asks to satisfy one bid (in-order)
        m.orders = m.orders[:0]
        m.accept_order(Order(-1,0,0,100,1), None)
        m.accept_order(Order(1,0,1,10,1), None)
        m.accept_order(Order(1,0,2,20,1), None)
        m.accept_order(Order(1,0,3,30,1), None)
        m.accept_order(Order(1,0,4,50,1), None)
        m.define_order_id()
        matches = m.match(m.get_order_dict())
        assert len(matches) == 4
        assert matches[0]["energy"] == 10
        assert matches[1]["energy"] == 20
        assert matches[2]["energy"] == 30
        assert matches[3]["energy"] == 40  # only 100 in bid
