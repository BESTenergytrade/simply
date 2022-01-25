from simply.market import Market
from simply.actor import Order

from argparse import ArgumentParser
import json

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


class PayAsBidMatchingAlgorithm():

    def get_matches_recommendations(mycoDict):

        recommendations = []

        for market_id, market in mycoDict.items():
            for time, orders in market.items():
                m = Market(time = time + ":00")
                bids = {bid["id"]: bid for bid in orders["bids"]}
                asks = {ask["id"]: ask for ask in orders["offers"]}

                accept_orders(m, orders)
                matches = m.match()

                recommendations += generate_recommendations(market_id, time, bids, asks, matches)

        return recommendations


if __name__ == "__main__":
    parser = ArgumentParser(description='Wrapper for myco API')
    parser.add_argument('file', help='myco API data file')
    args = parser.parse_args()

    with open(args.file, 'r') as f:
        mycoDict = json.load(f)

    recommendation = PayAsBidMatchingAlgorithm.get_matches_recommendations(mycoDict)
    print(json.dumps(recommendation, indent=2))
