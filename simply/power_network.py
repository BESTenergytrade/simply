import json
import random
import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt
from pathlib import Path

import simply.config as cfg


class PowerNetwork:
    """
    Representation of energy grid and associated grid fees.
    """

    def __init__(self, name, network, weight_factor=None):
        """
        New network model. Sets edge weights to leaf nodes to 0 (cluster).
        Calculates shortest paths, as network is unlikely to change again.

        :param name: name of network :type name: string :param network:
        graph representation :type network: networkx
        graph :param weight_factor: scale graph edge weights to power
        transmission cost. Can be set in config:
        network->weight_factor. Default 1 :type weight_factor: float
        """

        self.name = name
        # all leaf nodes are potential connection points for actors
        self.leaf_nodes = [n for n, d in network.degree() if d == 1]

        # holds shortest paths from each node to each other node
        self.short_paths = None
        # clusters is list of sets with node IDs. Unused.
        self.clusters = []
        # reverse lookup: node ID -> cluster index. Unused.
        self.node_to_cluster = {}
        # matrix with (scaled) weights between clusters
        self.grid_fee_matrix = []

        if weight_factor is None:
            weight_factor = cfg.parser.getfloat("network", "weight_factor", fallback=1)

        self.network = network
        self.update_shortest_paths()
        self.generate_grid_fee_matrix(weight_factor)

    def update_shortest_paths(self):
        self.short_paths = nx.shortest_path(self.network, weight="weight")

    def generate_grid_fee_matrix(self, weight_factor=1):
        # clustering of nodes by weight. Within cluster, edges have weight 0

        # Reset cluster list, lookup and fee matrix
        self.clusters = []
        self.node_to_cluster = {}
        self.grid_fee_matrix = []

        # BFS: start with any node
        nodes = [list(self.network.nodes)[0]]
        while nodes:
            # get first node from list. Guaranteed to not be part of prior cluster
            u = nodes.pop(0)
            # start new cluster with this node
            cluster = len(self.clusters)
            self.clusters.append({u})
            self.node_to_cluster[u] = cluster
            # check neighbors using BFS
            cluster_nodes = [u]
            while cluster_nodes:
                # get next neighbor node
                node = cluster_nodes.pop(0)
                for edge in self.network.edges(node, data=True):
                    # get target of this connection (neighbor of neighbor)
                    v = edge[1]
                    if v in self.node_to_cluster:
                        # already visited
                        continue
                    if edge[2].get("weight", 0) == 0:
                        # weight zero: part of cluster
                        # add to cluster set
                        self.clusters[-1].add(v)
                        self.node_to_cluster[v] = cluster
                        # add to list of neighbors to check later
                        cluster_nodes.append(v)
                    else:
                        # not part of cluster
                        # add to list of nodes that form new clusters
                        nodes.append(v)

        # Calculate accumulated weights on path between clusters and actor nodes
        # Get any one node from each cluster
        root_nodes = {i: list(c)[0] for i, c in enumerate(self.clusters)}
        # init weight matrix with zeros
        num_root_nodes = len(root_nodes)
        self.grid_fee_matrix = [[0] * num_root_nodes for _ in range(num_root_nodes)]
        # fill weight matrix
        # matrix symmetric: only need to compute half of values, diagonal is 0
        for i, n1 in root_nodes.items():
            for j, n2 in root_nodes.items():
                if i > j:
                    # get weight between n1 and n2
                    w = self.get_path_weight(n1, n2) * weight_factor
                    self.grid_fee_matrix[i][j] = w
                    self.grid_fee_matrix[j][i] = w
                if i == j:
                    self.grid_fee_matrix[i][j] = cfg.config.local_grid_fee
        print(f"Generated grid fee matrix: {self.grid_fee_matrix}")

    def to_image(self, dirpath=Path("./")):
        fig = self.plot(False)
        fig.savefig(dirpath / f"{self.name}.png")
        plt.close(fig)

    def plot(self, show=True):
        fig = plt.figure()
        # TODO: improved plot with or without Graphvis
        # from simply.plotting import plot_hierarchical
        # plot_hierarchical(self.network)
        try:
            plot_topology_graphvis(self.network)
        except ImportError:
            # In case dot is not installed
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
        actor_nodes = list([a.id for a in actors])
        random.shuffle(actor_nodes)
        connections = random.choices(self.leaf_nodes, k=len(actors))
        map = {}
        while actor_nodes:
            a = actor_nodes.pop()
            c = connections.pop()
            map[a] = c
            self.network.add_edge(c, a, weight=0)

        # The Actor knows its grid connection node ID
        for a in actors:
            a.grid_id = map[a.id]

        return map

    def add_actors_map(self, map):
        for a, c in map.items():
            self.network.add_edge(c, a, weight=0)

        return map

    def get_path_weight(self, u, v):
        # get weight of path between nodes
        # returns sum of edge weights on shortest path
        if u == v:
            return 0
        if self.short_paths is None:
            self.update_shortest_paths()

        path = self.short_paths[u][v]
        path_weight = 0
        u = path[0]
        for v in path[1:]:
            path_weight += self.network[u][v].get("weight", 0)
            u = v
        return path_weight

    def get_cluster_weights(self, c1, c2):
        # get all weighted pathlength between two clusters of nodes
        # returns dict with nodes from c1 -> nodes from c2 -> weight
        # unused
        weights = {u: {} for u in c1}
        for u in c1:
            for v in c2:
                weights[u][v] = self.get_path_weight(u, v)
        return weights


def create_random(nodes):
    nw = nx.random_tree(nodes)

    # Add random weights in [0, 1] with 0.1 resolution
    for e in nw.edges:
        nw[e[0]][e[1]]["weight"] = random.randint(0, 10) * 0.1

    return PowerNetwork("random", nw)


def create_random2(nodes):
    nw = nx.random_tree(nodes)
    # TODO check also: nx.balanced_tree(branches, height)

    leaf_nodes = sorted(n for n, d in nw.degree() if d == 1)
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
    # Add random weights in [0, 1] with 0.1 resolution
    for e in nw.edges:
        nw[e[0]][e[1]]["weight"] = random.randint(0, 10) * 0.1
    return PowerNetwork("random", nw)


def create_power_network_from_config(network_path, weight_factor=1):
    with open(network_path) as user_file:
        file_contents = user_file.read()
    network_json = json.loads(file_contents)
    network_name = list(network_json.keys())[0]
    network_json = list(network_json.values())[0]
    network = json_graph.node_link_graph(network_json,
                                         directed=network_json.get("directed", False),
                                         multigraph=network_json.get("multigraph", False))
    return PowerNetwork(network_name, network, weight_factor)


def remove_weights_from_leef_nodes(network):
    leaf_nodes = [n for n, d in network.degree() if d == 1]
    # leaves with their parents (no weight between them)
    for leaf in leaf_nodes:
        for u, v, d in network.edges(leaf, data=True):
            d["weight"] = 0


def plot_topology_graphvis(G):
    from networkx.drawing.nx_pydot import graphviz_layout
    pos = graphviz_layout(G, prog="dot")
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=50)
