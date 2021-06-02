from simply import scenario
from simply.market import Market

timesteps = 1

if __name__ == "__main__":
    sc = scenario.create_random(12, 10)
    print(sc.to_dict())

    for t in range(timesteps):
        m = Market()
        # TODO for loop over actors, which place bids and asks
        # TODO concurrent bidding of actors
        m.accept_bid((0, 0, 3, 0.2))
        m.accept_bid((0, 2, 2, 0.2))
        m.accept_ask((0, 1, 4, 0.2))

        m.print()
        # To run matching without effective clearing
        m.match(show=True)

        m.clear()
        print("Matches of bid/ask ids: {}".format(m.get_all_matches()))
