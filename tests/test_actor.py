import warnings

import pandas as pd
import numpy as np

from simply.actor import Actor, create_random, Order
from simply.battery import Battery
from simply.market_fair import BestMarket, MARKET_MAKER_THRESHOLD
from simply.power_network import PowerNetwork
import simply.config as cfg
import networkx as nx

from simply.scenario import Scenario


class TestActor:

    df = pd.DataFrame(np.random.rand(24, 4), columns=["load", "pv", "prices", "schedule"])
    cfg.Config("")
    nw = nx.Graph()
    nw.add_edges_from([(0, 1, {"weight": 1}), (1, 2), (1, 3), (0, 4)])
    pn = PowerNetwork("", nw, weight_factor=1)
    test_prices = [0.082,0.083,0.087,0.102,0.112,0.122,0.107,0.103,0.1,0.1,0.09,0.082,0.083,0.083,0.094,0.1,0.11,0.109,0.106,0.105,0.1,0.093,0.084,0.081,0.078,0.074,0.074,0.079,0.081,0.083,0.079,0.074,0.07,0.067,0.065,0.067,0.073,0.075,0.085,0.095,0.107,0.107,0.107,0.107,0.1,0.094,0.087,0.08,]

    test_schedule = [0.164,0.077,0.019,-0.038,-0.281,-0.054,-0.814,-1.292,-1.301,-1.303,-1.27,-1.228,-1.301,-0.392,-0.564,-0.411,0.046,0.385,0.448,0.553,0.143,0.123,0.172,0.094,0.084,0.075,-0.017,-0.071,-0.147,-0.23,-1.208,-1.072,-0.277,-0.274,-0.813,-0.131,-0.844,0.013,0.071,-0.027,-0.005,0.08,0.065,0.406,0.403,0.341,0.096,0.098,]
    example_df = pd.DataFrame(list(zip([abs(num) for num in test_schedule],
                                       [0 for i in range(len(test_schedule))], test_schedule,
                                       test_prices)),
                              columns=['load', 'pv', 'schedule', 'prices'])

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
        pn = PowerNetwork("", nx.random_tree(1))

        scenario = Scenario(pn, [], None,steps_per_hour=4)
        battery = Battery(capacity=3, max_c_rate=2, soc_initial=0.0)
        actor = Actor(0, self.example_df, battery=battery, scenario=scenario)

        m = BestMarket(0, self.pn)
        # Iterate through timesteps
        for t in range(20):
            m.t = t
            actor.get_market_schedule(strategy=1)
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
        # Iterate through time steps
        for t in range(20):
            m.t = t
            actor.get_market_schedule(strategy=2)
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
        a.get_market_schedule(strategy=1)
        assert a.market_schedule.__abs__().sum() == 0
        a.get_market_schedule(strategy=3)
        assert a.market_schedule.__abs__().sum() > 0
        o = a.generate_order()
        assert len(a.orders) == 1
        assert o.actor_id == 0


    def test_rule_based_strategy_3_fixed(self):
        bat = Battery(max_c_rate=2, soc_initial=0, capacity=13.5)
        a = Actor(0, self.example_df, battery=bat)
        a.data.selling_price = a.data.prices
        bank = 0
        actual_market_schedule = []
        socs = []
        prices = []
        sched=[]
        for time in range(24):
            a.create_prediction()
            a.get_market_schedule(strategy=2)
            # delta_energy= a.market_schedule[0]-a.pred.schedule.iloc[0]
            # bat.get_energy(delta_energy)
            a.update()
            a.t +=1
            bank += a.market_schedule[0]*a.data.prices[0]
            actual_market_schedule.append(a.market_schedule[0])
            socs.append(a.battery.soc)
            prices.append(a.pred.prices[0]*40)
            sched.append(a.pred.schedule[0])
            if time == 0:
                first_market_schedule= a.market_schedule.copy()
                first_pred_soc = a.predicted_soc.copy()
            #
        df = pd.DataFrame(actual_market_schedule, columns=["market_schedule"])

        # df["predicted_soc"] = first_pred_soc
        df["socs"] = socs
        df["prices"] = prices
        df["schedule"] = sched
        # df["actual_market"]=actual_market_schedule
        df.plot()
        print(f"Bank: {bank}")

with warnings.catch_warnings() as w:
    # Cause all warnings to always be triggered.
    warnings.filterwarnings("ignore", category=FutureWarning)
    t = TestActor()
    t.test_rule_based_strategy_3_fixed()