from simply import market, market_2pac, market_fair
from simply.actor import Order

from argparse import ArgumentParser
import json
from abc import ABC, abstractmethod

from simply.config import Config

# default config
Config('', '')

ENERGY_UNIT_CONVERSION_FACTOR = 1  # simply: kW


def accept_orders(market, time, orders):
    # generate simply Order, put it into market including predefined order IDs
    # apply conversion factor except for market maker orders
    for bid in orders["bids"]:
        energy = min(bid["energy"] * ENERGY_UNIT_CONVERSION_FACTOR, 2 ** 63 - 1)
        cluster = bid.get("cluster")
        order = Order(-1, time, bid["buyer"], cluster, energy, bid["energy_rate"])
        market.accept_order(order, order_id=bid["id"])
    for ask in orders["offers"]:
        energy = min(ask["energy"] * ENERGY_UNIT_CONVERSION_FACTOR, 2 ** 63 - 1)
        cluster = ask.get("cluster")
        order = Order(1, time, ask["seller"], cluster, energy, ask["energy_rate"])
        market.accept_order(order, order_id=ask["id"])


def generate_recommendations(market_id, time, bids, asks, matches):
    recommendations = []

    for match in matches:
        recommendations.append({
            "market_id": market_id,
            "time_slot": time,
            'matching_requirements': None,
            "bid": bids[match["bid_id"]],
            "offer": asks[match["ask_id"]],
            "selected_energy": match["energy"] / ENERGY_UNIT_CONVERSION_FACTOR,
            "trade_rate": match["price"],
        })

    return recommendations


class MatchingAlgorithm(ABC):
    @staticmethod
    def get_market_matches(mycoDict, market, grid_fee_matrix=None):
        """
        Unpacks order dictionary per market and time slot
        and match the orders using the given market.

        :param mycoDict: hierarchical dictionary with market name and time slot each containing a
            dict with bids and offers in lists {'bids': [], 'offers': []}
        :type mycoDict: dict
        :param market: Market object that implements the matching algorithm
        :param grid_fee_matrix: two-dimensional nXn list used to calculate grid-fees e.g.,
            [[0,1],[1,0]]
        :return: list of dictionaries with matches in all given markets and time slots
        """
        # ToDo in doc above Market object? Is it not a Market constructor?

        recommendations = []
        # Market is coupled with scenario and environment data. Market looks up time in environment.

        for market_id, market_name in mycoDict.items():
            for time, orders in market_name.items():
                m = market(grid_fee_matrix=grid_fee_matrix, time_step=time)
                bids = {bid["id"]: bid for bid in orders["bids"]}
                asks = {ask["id"]: ask for ask in orders["offers"]}

                accept_orders(m, time, orders)
                matches = m.match()

                recommendations += generate_recommendations(market_id, time, bids, asks,
                                                            matches)

        return recommendations

    @abstractmethod
    def get_matches_recommendations(cls, mycoDict):
        pass


class BestPayAsBidMatchingAlgorithm(MatchingAlgorithm):
    """
    Wrapper class for the pay as bid matching algorithm
    """

    @classmethod
    def get_matches_recommendations(cls, mycoDict, grid_fee_matrix=None):
        return super().get_market_matches(mycoDict, market.Market, grid_fee_matrix)


class BestPayAsClearMatchingAlgorithm(MatchingAlgorithm):
    """
    Wrapper class for the pay as clear matching algorithm
    """

    @classmethod
    def get_matches_recommendations(cls, mycoDict, grid_fee_matrix=None):
        return super().get_market_matches(mycoDict, market_2pac.TwoSidedPayAsClear, grid_fee_matrix)


class BestClusterPayAsClearMatchingAlgorithm(MatchingAlgorithm):
    """
    Wrapper class of the cluster-based market fair matching algorithm
    """

    @classmethod
    def get_matches_recommendations(cls, mycoDict, grid_fee_matrix=None):
        return super().get_market_matches(mycoDict, market_fair.BestMarket, grid_fee_matrix)


if __name__ == "__main__":
    parser = ArgumentParser(description='Wrapper for myco API')
    parser.add_argument('file', help='myco API data file')
    args = parser.parse_args()

    with open(args.file, 'r') as f:
        mycoDict = json.load(f)

    recommendation = BestPayAsBidMatchingAlgorithm.get_matches_recommendations(mycoDict)
    # recommendation = BestPayAsClearMatchingAlgorithm.get_matches_recommendations(mycoDict)
    # recommendation = BestClusterPayAsClearMatchingAlgorithm.get_matches_recommendations(mycoDict)
    print(json.dumps(recommendation, indent=2))
