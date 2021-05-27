import json
import random
import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt


class PowerNetwork:
    def __init__(self, name, network):
        self.name = name
        self.network = network
        # TODO currently first leaf node is interpreted as root node which might change
        self.leaf_nodes = sorted(n for n, d in self.network.degree() if d == 1)[1:]
        self.short_paths = {}

    def update_shortest_paths(self):
        self.short_paths = dict(nx.all_pairs_shortest_path(self.network))

    def to_image(self, width, height):
        fig = self.plot(False)
        fig.savefig(self.name + ".png")

    def plot(self, show=True):
        fig = plt.figure()
        # TODO: improved plot with or without Graphvis
        # from simply.plotting import plot_hierarchical
        # plot_hierarchical(self.network)
        nx.draw(self.network, with_labels=True, font_weight="bold", node_size=50)
        if show:
            plt.show()

        return fig

    def to_dict(self):
        return json_graph.node_link_data(self.network)

    def to_json(self):
        filename = self.name + ".json"
        net_json = json_graph.node_link_data(self.network)
        json.dump(net_json, open(filename, "w"), indent=2)

    def add_actors_random(self, actors):
        base_id = max(self.network.nodes()) + 100
        num_actors = len(actors)
        actor_nodes = list([*range(base_id, base_id + len(actors), 1)])
        random.shuffle(actor_nodes)
        connections = random.choices(self.leaf_nodes, k=num_actors)
        map = {}
        while actor_nodes:
            a = actor_nodes.pop()
            c = connections.pop()
            map[a] = c
            self.network.add_edge(c, a)

        return map

    def add_actors_map(self, map):
        base_id = max(self.network.nodes()) + 100
        for a, c in map.items():
            self.network.add_edge(c, a + base_id)

        return map


def create_random(nodes):
    # TODO get number of leaves as parameter
    nw = nx.random_tree(nodes)

    # Add random weights in [0, 1] with 0.1 resolution
    for e in nw.edges:
        nw[e[0]][e[1]]["weight"] = random.randint(0, 10) * 0.1

    return PowerNetwork("random", nw)


def create_random2(nodes):
    # TODO get number of leaves as parameter
    nw = nx.random_tree(nodes)

    # TODO might be useful to add actors as additional nodes after creation
    leaf_nodes = sorted(n for n, d in nw.degree() if d == 1)[1:]
    offset = 0
    for n in leaf_nodes:
        for i in range(random.randint(2, 5)):
            offset += 1
            nw.add_edge(n, nodes + offset)

    # Add random weights in [0, 1] with 0.1 resolution
    for e in nw.edges:
        nw[e[0]][e[1]]["weight"] = random.randint(0, 10) * 0.1

    return PowerNetwork("random", nw)


def load_network():
    nw = nx.random_lobster(3, 0.5, 0.2)
    return PowerNetwork("random", nw)
