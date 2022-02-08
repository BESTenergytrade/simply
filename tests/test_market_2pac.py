from simply.actor import Order
from simply.market_2pac import TwoSidedPayAsClear

import pytest

def test_basic():
    m = TwoSidedPayAsClear(0)
    # no orders: no matches
    matches = m.match()
    assert len(matches) == 0

    # only one type: no match
    m.accept_order(Order(-1,0,0,None,1,1), None)
    matches = m.match()
    assert len(matches) == 0

    # bid and ask with same energy and price
    m.accept_order(Order(1,0,1,None,1,1), None)
    matches = m.match()
    assert len(matches) == 1
    # check match
    assert matches[0]["time"] == 0
    assert matches[0]["bid_actor"] == 0
    assert matches[0]["ask_actor"] == 1
    assert matches[0]["energy"] == 1
    assert matches[0]["price"] == 1

def test_prices():
    # different prices
    m = TwoSidedPayAsClear(0)
    # ask above bid: no match
    m.accept_order(Order(-1,0,0,None,1,2), None)
    m.accept_order(Order(1,0,1,None,1,2.5), None)
    matches = m.match()
    assert len(matches) == 0

    # reset orders
    m.orders = m.orders[:0]
    # ask below bid: take lower one
    m.accept_order(Order(-1,0,0,None,1,2.5), None)
    m.accept_order(Order(1,0,1,None,1,2), None)
    matches = m.match()
    assert len(matches) == 1
    assert matches[0]["energy"] == 1
    assert matches[0]["price"] == 2

def test_energy():
    # different energies
    m = TwoSidedPayAsClear(0)
    m.accept_order(Order(-1,0,0,None,.1,1), None)
    m.accept_order(Order(1,0,1,None,1,1), None)
    matches = m.match()
    assert len(matches) == 1
    assert matches[0]["energy"] == 0.1

    m.orders = m.orders[:0]
    m.accept_order(Order(-1,0,0,None,100,1), None)
    m.accept_order(Order(1,0,1,None,.3,1), None)
    matches = m.match()
    assert len(matches) == 1
    assert matches[0]["energy"] == pytest.approx(0.3)

def test_setting_order_id():
    # Check if matched orders retain original ID
    m = TwoSidedPayAsClear(0)
    m.accept_order(Order(-1,0,2,.2,1), None, "ID1")
    m.accept_order(Order(1,0,3,1,1), None, "ID2")
    matches = m.match()
    assert len(matches) == 1
    assert matches[0]["energy"] == pytest.approx(0.2)
    assert matches[0]["bid_id"] == "ID1"
    assert matches[0]["ask_id"] == "ID2"

def test_multiple():
    # multiple bids to satisfy one ask
    m = TwoSidedPayAsClear(0)
    m.accept_order(Order(-1, 0, 1, None, 11, 1.1), None)
    m.accept_order(Order(-1,0,0,None,.1,3), None)
    m.accept_order(Order(1,0,2,None,2,1), None)
    matches = m.match()
    assert len(matches) == 2
    assert matches[0]["energy"] == 0.1
    assert matches[1]["energy"] == 1.9  # only 2 in ask

    # multiple asks to satisfy one bid (order by price)
    m.orders = m.orders[:0]
    m.accept_order(Order(1, 0, 3, None, 30, 3), None)
    m.accept_order(Order(1, 0, 4, None, 50, 4), None)
    m.accept_order(Order(-1, 0, 0, None, 100, 5), None)
    m.accept_order(Order(1,0,1,None,10,1), None)
    m.accept_order(Order(1,0,2,None,20,2), None)

    matches = m.match()
    assert len(matches) == 4
    assert matches[0]["energy"] == 10
    assert matches[1]["energy"] == 20
    assert matches[2]["energy"] == 30
    assert matches[3]["energy"] == 40  # only 100 in bid
