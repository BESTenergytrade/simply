import networkx


class PowerNetwork:
    def __init__(self, network):
        self.network = network

    def to_image(self, width, height):
        pass

    def plot(self):
        pass

    def to_dict():
        pass


def create_random():
    # TODO create network using a method that guarantees a certain number of nodes (not random_lobster)
    # TODO get number of leaves as parameter
    # TODO add weights for computing the energy loss
    nw = networkx.random_lobster(3, 0.5, 0.2)
    return PowerNetwork(nw)
