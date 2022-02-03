from simply import market, market_2pac, market_fair
from simply.actor import Order
from simply import power_network

from argparse import ArgumentParser
import json
from abc import ABC, abstractmethod

from simply.config import Config
# default config
Config('')


ENERGY_UNIT_CONVERSION_FACTOR = 1000  # simply: kW, D3A: MW


def accept_orders(market, orders):
    # generate simply Order, put it into market
    # apply conversion factor except for market maker orders
    for bid in orders["bids"]:
        energy = min(bid["energy"] * ENERGY_UNIT_CONVERSION_FACTOR, 2**63-1)
        order = Order(-1, bid["time_slot"], bid["id"], energy, bid["energy_rate"])
        market.accept_order(order, None)
    for ask in orders["offers"]:
        energy = min(ask["energy"] * ENERGY_UNIT_CONVERSION_FACTOR, 2**63-1)
        order = Order(1, ask["time_slot"], ask["id"], energy, ask["energy_rate"])
        market.accept_order(order, None)


def generate_recommendations(market_id, time, bids, asks, matches):
    recommendations = []

    for match in matches:
        recommendations.append({
            "market_id": market_id,
            "time_slot": time,
            "bids": [bids[match["bid_actor"]]],
            "offers": [asks[match["ask_actor"]]],
            "selected_energy": match["energy"] / ENERGY_UNIT_CONVERSION_FACTOR,
            "trade_rate": match["price"],
        })

    return recommendations


class MatchingAlgorithm(ABC):
    @staticmethod
    def get_market_matches(mycoDict, market):
        recommendations = []

        for market_id, market_name in mycoDict.items():
            for time, orders in market_name.items():
                m = market(time=time+":00")
                bids = {bid["id"]: bid for bid in orders["bids"]}
                asks = {ask["id"]: ask for ask in orders["offers"]}

                accept_orders(m, orders)
                matches = m.match()

                recommendations += generate_recommendations(market_id, time, bids, asks, matches)

        return recommendations

    @abstractmethod
    def get_matches_recommendations(cls, mycoDict):
        pass


class PayAsBidMatchingAlgorithm(MatchingAlgorithm):
    @classmethod
    def get_matches_recommendations(cls, mycoDict):
        return super().get_market_matches(mycoDict, market.Market)


class PayAsClearMatchingAlgorithm(MatchingAlgorithm):
    @classmethod
    def get_matches_recommendations(cls, mycoDict):
        return super().get_market_matches(mycoDict, market_2pac.TwoSidedPayAsClear)


class ClusterPayAsClearMatchingAlgorithm(MatchingAlgorithm):
    @classmethod
    def get_matches_recommendations(cls, mycoDict):
        pn = power_network.create_random(1)
        recommendations = []

        for market_id, market_name in mycoDict.items():
            for time, orders in market_name.items():
                actors = [bid["id"] for bid in orders["bids"]] + \
                         [ask["id"] for ask in orders["offers"]]
                # Give actors a position in the network
                # Currently at a single node with id 0
                actor_nodes = [0 for i in actors]
                map_actors = {actor: node_id for actor, node_id in zip(actors, actor_nodes)}
                pn.add_actors_map(map_actors)

                m = market_fair.BestMarket(t=time + ":00", network=pn)
                bids = {bid["id"]: bid for bid in orders["bids"]}
                asks = {ask["id"]: ask for ask in orders["offers"]}

                accept_orders(m, orders)
                matches = m.match()

                recommendations += generate_recommendations(market_id, time, bids, asks,
                                                            matches)

        return recommendations


if __name__ == "__main__":
    parser = ArgumentParser(description='Wrapper for myco API')
    parser.add_argument('file', help='myco API data file')
    args = parser.parse_args()

    with open(args.file, 'r') as f:
        mycoDict = json.load(f)

    # recommendation = PayAsBidMatchingAlgorithm.get_matches_recommendations(mycoDict)
    # recommendation = PayAsClearMatchingAlgorithm.get_matches_recommendations(mycoDict)
    recommendation = ClusterPayAsClearMatchingAlgorithm.get_matches_recommendations(mycoDict)
    print(json.dumps(recommendation, indent=2))
