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

    def to_dict():
        return {
            'power_network': self.power_network.to_dict(),
            'actors': self.actors.to_dict(),
            'map_actors': self.map_actors.to_dict(),
        }

    def from_dict():
        pass


def create_random(num_actors):
    pn = power_network.create_random()
    actors = [actor.create_random(i) for i in range(num_actors)]

    # Give actors a random position in the network
    # TODO one node could also contain several actors (using random.choices method)
    actor_nodes = random.sample(pn.network.nodes, num_actors)

    map_actors = {actor.id: node_id for actor, node_id in zip(actors, actor_nodes)}

    return Scenario(pn, actors, map_actors)
