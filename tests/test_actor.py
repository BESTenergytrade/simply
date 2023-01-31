import pandas as pd
import numpy as np

from simply.actor import Actor, create_random
from simply.battery import Battery
from simply.market_fair import BestMarket


class TestActor:
    df = pd.DataFrame(np.random.rand(24, 4), columns=["load", "pv", "prices", "schedule"])

    def test_init(self):
        # actor_id, dataframe, load_scale, power_scale, pm (?)
        a = Actor(0, self.df)
        assert a is not None

    def test_generate_orders(self):
        a = Actor(0, self.df)
        o = a.generate_order()
        assert len(a.orders) == 1
        assert o.actor_id == 0

    def test_recv_market_results(self):
        # time, sign, energy, price
        a = Actor(0, self.df)
        o = a.generate_order()
        assert o.time == 0
        a.receive_market_results(o.time, o.type, o.energy / 4, o.price)
        # must be tuple at key 0 (current time)
        assert len(a.traded[0]) == 2
        # must have one energy and one price
        assert len(a.traded[0][0]) == 1
        assert len(a.traded[0][1]) == 1
        # double matches should not overwrite each other
        a.receive_market_results(o.time, o.type, o.energy / 2, o.price)
        # must still be tuple
        assert len(a.traded[0]) == 2
        # tuple must now contain lists of len 2
        assert len(a.traded[0][0]) == 2
        assert len(a.traded[0][1]) == 2

    def test_create_random(self):
        a = create_random(0)
        assert a.id == 0

    def test_create_random_multidays_halfhour(self):
        data_cfg = {"nb_ts": 96, "ts_hour": 2}
        a = create_random(0, **data_cfg)
        # time series is longer than one day
        assert a.data.index[0].date() != a.data.index[-1].date()

    def test_rule_based_strategy_1(self):
        prices = [0.2116079447, 0.1473127859, 0.22184087530000002, 0.11761082760000001,
                  0.2463169965,
                  0.2020745841, 0.0613031114, 0.24701460990000002, 0.12690570210000002,
                  0.1467477666,
                  0.0910571313, 0.1510937983, 0.0961995166, 0.16232426160000002, 0.1911430976,
                  0.2395885052,
                  0.1161007245, 0.1912644558, 0.08394693780000001, 0.031559975000000004,
                  0.07516904740000001, 0.0839614066, 0.1340712662, 0.1921131123]

        schedule = [-0.2278688066, -0.4956801147, -0.5660800508, -0.4605807878, -0.7235523078,
                    -0.41539310830000004, -0.0517064662, -0.4741886065, -0.253179973,
                    -0.7055324580000001,
                    -0.0665372924, -0.33647962400000003, -0.3992714075, -0.4354996278,
                    -0.625752089,
                    -0.30241824170000003, -0.23024240310000002, -0.6122942333, -0.1880810302,
                    -0.1261036003,
                    -0.18803270630000002, -0.2284269156, -0.7287319187, -0.0596583833]

        df = pd.DataFrame(list(zip([abs(num) for num in schedule], [0 for i in range(len(
            schedule))], schedule, prices)), columns=['load', 'pv', 'schedule', 'prices'])
        battery = Battery(capacity=3, max_c_rate=2, soc_initial=0.0)
        actor = Actor(0, df, battery=battery)
        m = BestMarket(0, self.pn)
        # Iterate through timesteps
        for _ in range(30):
            actor.plan_global_supply()
            actor.generate_order()
            # Generate market maker order
            # m.accept_order(-1, 0, 3, 1, actor.global_market_buying_plan[0],
            #                actor.pred.global_price[0])
            # Generate market maker order
            m.accept_order(Order(1, 0, 'market_maker', None, MARKET_MAKER_THRESHOLD, 1))
            matches = m.match()

    def test_rule_based_strategy_2(self):
        pass

    def test_rule_based_strategy_3(self):
        pass
