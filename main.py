import pandas as pd

from simply import scenario
from simply.market import Market

timesteps = 2

if __name__ == "__main__":
    sc = scenario.create_random(12, 10)
    print(sc.to_dict())

    for t in range(timesteps):
        m = Market(t)
        for a in sc.actors:
            # TODO concurrent bidding of actors
            # TODO integrate order type into order
            order, o_type = a.generate_order()
            if o_type == -1:
                m.accept_bid(order, a.receive_market_results)
            elif o_type == 1:
                m.accept_ask(order, a.receive_market_results)
            else:
                raise ValueError

        m.print()
        # To run matching without effective clearing
        m.match(show=True)

        m.clear()
        print("Matches of bid/ask ids: {}".format(m.get_all_matches()))

        print("Check traded energy volume and price")
        print(pd.DataFrame.from_dict([a.traded for a in sc.actors]).unstack().apply(pd.Series))
