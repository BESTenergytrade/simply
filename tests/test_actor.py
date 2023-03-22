import warnings

import pandas as pd
import numpy as np

from simply.util import NoNextBuyException
from simply.actor import Actor, create_random, Order
from simply.battery import Battery
from simply.market_fair import BestMarket, MARKET_MAKER_THRESHOLD
from simply.power_network import PowerNetwork
import simply.config as cfg
import networkx as nx


class TestActor:
    df = pd.DataFrame(np.random.rand(24, 4), columns=["load", "pv", "prices", "schedule"])
    cfg.Config("")
    nw = nx.Graph()
    nw.add_edges_from([(0, 1, {"weight": 1}), (1, 2), (1, 3), (0, 4)])
    pn = PowerNetwork("", nw, weight_factor=1)
    test_prices = [0.2116079447, 0.1473127859, 0.22184087530000002, 0.11761082760000001,
                   0.2463169965, 0.2020745841, 0.0613031114, 0.24701460990000002,
                   0.12690570210000002, 0.1467477666, 0.0910571313, 0.1510937983, 0.0961995166,
                   0.16232426160000002, 0.1911430976, 0.2395885052, 0.1161007245, 0.1912644558,
                   0.08394693780000001, 0.031559975000000004, 0.07516904740000001, 0.0839614066,
                   0.1340712662, 0.1921131123]

    test_schedule = [-0.2278688066, -0.4956801147, -0.5660800508, -0.4605807878, -0.7235523078,
                     -0.41539310830000004, -0.0517064662, -0.4741886065, -0.253179973,
                     -0.7055324580000001, -0.0665372924, -0.33647962400000003, -0.3992714075,
                     -0.4354996278, -0.625752089, -0.30241824170000003, -0.23024240310000002,
                     -0.6122942333, -0.1880810302, -0.1261036003, -0.18803270630000002,
                     -0.2284269156, -0.7287319187, -0.0596583833]
    example_df = pd.DataFrame(list(zip([abs(num) for num in test_schedule],
                                       [0 for i in range(len(test_schedule))], test_schedule,
                                       test_prices)),
                              columns=['load', 'pv', 'schedule', 'prices'])

    def test_init(self):
        # actor_id, dataframe, load_scale, power_scale, pm (?)
        a = Actor(0, self.df)
        assert a is not None

    def test_generate_orders(self):
        bat = Battery(2, soc_initial=0)
        a = Actor(0, self.df, battery=bat)
        a.market_schedule *= 0
        try:
            o = a.generate_order()
        except NoNextBuyException:
            pass
        else:
            raise Exception("Order was generated although no market schedule is only Zeros")

        # Make sure there is load in the schedule so buying power is necessary
        a.data.loc[0:5, "schedule"] = -a.data.loc[0:5, "schedule"].abs()
        a.update()
        a.generate_market_schedule(strategy=1)
        o = a.generate_order()
        assert len(a.orders) == 1
        assert o.actor_id == 0

    def test_recv_market_results(self):
        # time, sign, energy, price
        bat = Battery(2, soc_initial=0)
        a = Actor(0, self.df, battery=bat)
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
        battery = Battery(capacity=3, max_c_rate=2, soc_initial=0.0)
        actor = Actor(0, self.example_df, battery=battery)
        m = BestMarket(0, self.pn)
        # Iterate through timesteps
        for t in range(20):
            m.t = t
            actor.generate_market_schedule(strategy=1)
            order = actor.generate_order()
            # Generate market maker order
            m.accept_order(order)
            # Generate market maker order
            m.accept_order(Order(1, t, 'market_maker', None, MARKET_MAKER_THRESHOLD, order.price))
            m.clear()
        assert len(m.matches) == 20

    def test_rule_based_strategy_2(self):
        battery = Battery(capacity=3, max_c_rate=2, soc_initial=0.0)
        actor = Actor(0, self.example_df, battery=battery)
        m = BestMarket(0, self.pn)
        # Iterate through timesteps
        for t in range(20):
            m.t = t
            actor.generate_market_schedule(strategy=2)
            order = actor.generate_order()
            # Generate market maker order
            m.accept_order(order)
            # Generate market maker order
            m.accept_order(Order(1, t, 'market_maker', None, MARKET_MAKER_THRESHOLD, order.price))
            m.clear()
        assert len(m.matches) == 20

    def test_rule_based_strategy_3(self):
        bat = Battery(max_c_rate=2, soc_initial=0, capacity=13.5)
        a = Actor(0, self.example_df, battery=bat)
        a.data.selling_price *= 0.8
        a.create_prediction()

        a.market_schedule *= 0
        a.generate_market_schedule(strategy=1)
        assert a.market_schedule.__abs__().sum() == 0
        a.generate_market_schedule(strategy=3)
        assert a.market_schedule.__abs__().sum() > 0
        o = a.generate_order()
        assert len(a.orders) == 1
        assert o.actor_id == 0


    def test_rule_based_strategy_3_fixed(self):
        bat = Battery(max_c_rate=2, soc_initial=0, capacity=13.5)
        a = Actor(0, self.example_df, battery=bat)
        a.data.selling_price *= 0.8

        for time in range(50):
            a.create_prediction()
            a.generate_market_schedule(strategy=3)
            # delta_energy= a.market_schedule[0]-a.pred.schedule.iloc[0]
            # bat.get_energy(delta_energy)
            a.update()
            a.t +=1

            print(bat.soc , a.predicted_soc[0])

            df = pd.DataFrame(a.market_schedule, columns=["market_schedule"])
            df["predicted_soc"] = a.predicted_soc
            df.plot()

with warnings.catch_warnings() as w:
    # Cause all warnings to always be triggered.
    warnings.filterwarnings("ignore", category=FutureWarning)
    t = TestActor()
    t.test_rule_based_strategy_3_fixed()