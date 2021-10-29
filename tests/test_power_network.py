import networkx as nx

from simply.power_network import PowerNetwork, create_random

class TestPowerNetwork:

    def test_init(self):
        # name, network
        network1 = nx.random_tree(1)
        pn = PowerNetwork("test", network1)
        assert pn.name == "test"

        pn = PowerNetwork("", network1)
        assert len(pn.network.nodes) == 1
        # only root: no leaves
        assert len(pn.leaf_nodes) == 0

        network2 = nx.random_tree(2)
        pn = PowerNetwork("", network2)
        assert len(pn.network.nodes) == 2
        assert len(pn.leaf_nodes) == 1

    def test_short_paths(self):
        network = nx.random_tree(2)
        network[0][1]["weight"] = 1
        pn = PowerNetwork("", network)
        pn.update_shortest_paths()
        assert pn.short_paths[0] == {0: [0], 1: [0, 1]}
        assert pn.short_paths[1] == {0: [1, 0], 1: [1]}

    def test_to_dict(self):
        network = nx.random_tree(2)
        network[0][1]["weight"] = 1
        pn = PowerNetwork("", network)
        d = pn.to_dict()
        # print(d)
        assert not d["directed"]
        assert d["nodes"] == [{'id': 0}, {'id': 1}]
        assert d["links"] == [{'weight': 1, 'source': 0, 'target': 1}]

    def test_create_random(self):
        pn = create_random(2)
        assert len(pn.network.nodes) == 2
        assert len(pn.network.edges) == 1
        assert 0 <= pn.network[0][1]["weight"] <= 1
