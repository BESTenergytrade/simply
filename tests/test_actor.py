import pandas as pd
import numpy as np

from simply.actor import Actor, create_random

class TestActor:
    df = pd.DataFrame(np.random.rand(24,3), columns = ["load", "pv", "prices"])

    def test_init(self):
        # actor_id, dataframe, load_scale, power_scale, pm (?)
        a = Actor(0, self.df)

    def test_generate_orders(self):
        a = Actor(0, self.df)
        o = a.generate_order()
        assert len(a.orders) == 1
        assert o.actor_id == 0

    def test_recv_market_results(self):
        # time, sign, energy, price
        a = Actor(0, self.df)
        o = a.generate_order()
        a.receive_market_results(o.time, o.type, o.energy/4, o.price)
        assert len(a.traded[0]) == 1
        # double matches should not overwrite each other
        a.receive_market_results(o.time, o.type, o.energy/2, o.price)
        assert len(a.traded[0]) == 2

    def test_create_random(self):
        a = create_random(0)
        assert a.id == 0
