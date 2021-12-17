from simply.market import Market
from simply.actor import Order

from argparse import ArgumentParser
import json

from simply.config import Config
# default config
Config('')


class PayAsBidMatchingAlgorithm():

    def get_matches_recommendations(mycoDict):

        recommendation = []

        for market_id, market in mycoDict.items():
            for time, orders in market.items():
                m = Market(time = time + ":00")
                m.energy_unit = 0.0005
                bids = {}
                asks = {}

                for bid in orders["bids"]:
                    order = Order(-1, bid["time_slot"], bid["id"], bid["energy"], bid["energy_rate"])
                    m.accept_order(order, None)
                    bids[bid["id"]] = bid
                for ask in orders["offers"]:
                    order = Order(1, ask["time_slot"], ask["id"], ask["energy"], ask["energy_rate"])
                    m.accept_order(order, None)
                    asks[ask["id"]] = ask

                matches = m.match()

                for match in matches:
                    recommendation.append({
                        "market_id": market_id,
                        "time_slot": time,
                        "bids": bids[match["bid_actor"]],
                        "offers": asks[match["ask_actor"]],
                        "selected_energy": match["energy"],
                        "trade_rate": match["price"],
                    })

        return recommendation


if __name__ == "__main__":
    parser = ArgumentParser(description='Wrapper for myco API')
    parser.add_argument('file', help='myco API data file')
    args = parser.parse_args()

    with open(args.file, 'r') as f:
        mycoDict = json.load(f)

    recommendation = PayAsBidMatchingAlgorithm.get_matches_recommendations(mycoDict)
    print(json.dumps(recommendation, indent=2))
