import random

from simply import actor
from simply import power_network


class Scenario:
    def __init__(self, network, actors, map_actors):
        self.power_network = network
        self.actors = list(actors)
        # maps node ids to actors
        self.map_actors = map_actors

    def from_config(path):
        pass

    def __str__(self):
        return "Scenario(network: {}, actors: {}, map_actors: {})".format(
            self.power_network, self.actors, self.map_actors
        )

    def to_dict(self):
        return {
            "power_network": self.power_network.to_dict(),
            "actors": {a.to_dict() for a in self.actors},
            "map_actors": self.map_actors,
        }

    def from_dict(self):
        pass


def create_random(num_nodes, num_actors):
    pn = power_network.create_random(num_nodes)
    actors = [actor.create_random(i) for i in range(num_actors)]

    # Add actor nodes at random position in the network
    # One network node can contain several actors (using random.choices method)
    map_actors = pn.add_actors_random(actors)

    return Scenario(pn, actors, map_actors)


def create_random2(num_nodes, num_actors):
    # num_actors has to be much smaller than num_nodes
    pn = power_network.create_random(num_nodes)
    actors = [actor.create_random(i) for i in range(num_actors)]

    # Give actors a random position in the network
    actor_nodes = random.sample(pn.leaf_nodes, num_actors)
    map_actors = {actor.id: node_id for actor, node_id in zip(actors, actor_nodes)}

    # TODO tbd if actors are already part of topology ore create additional nodes
    # pn.add_actors_map(map_actors)

    return Scenario(pn, actors, map_actors)
