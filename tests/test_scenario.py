import networkx as nx

from simply.scenario import Scenario, create_random
from simply.power_network import PowerNetwork


class TestScenario:

    def test_init(self):
        # network, actors, map_actors, market, buy_prices [rng_seed]
        pn = PowerNetwork("", nx.random_tree(1))
        # random seed, but None
        s = Scenario(pn, None, [])
        assert s.rng_seed is not None
        s = Scenario(pn, None, [], 0)
        assert s.rng_seed == 0

    def test_to_dict(self):
        pn = PowerNetwork("", nx.random_tree(1))
        s = Scenario(pn, None, [])
        assert set(s.to_dict()).issuperset({"rng_seed", "power_network", "actors", "map_actors"})

    def test_random(self):
        # num nodes, num actors
        num_actors = 2
        s = create_random(3, num_actors, 1)
        # actors are generated plus 1 market_maker
        assert len(s.market_participants) == num_actors + 1
        assert len(s.map_actors) == 2
        assert len(s.power_network.network.nodes) == 3 + 2
