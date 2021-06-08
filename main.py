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
            order = a.generate_order()
            m.accept_order(order, a.receive_market_results)

        m.print()

        m.clear()
        print("Matches of bid/ask ids: {}".format(m.get_all_matches()))

        print("\nCheck traded energy volume and price at actor level")
        print(
            pd.DataFrame.from_dict([a.traded for a in sc.actors])
            .unstack()
            .apply(pd.Series)
        )
        print("\nCheck traded energy volume and price at market level")
        print(m.trades)
