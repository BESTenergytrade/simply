from simply import scenario
from simply.scenario import Scenario
from simply.market import Market

timesteps = 1

if __name__ == "__main__":
    sc = scenario.create_random(12, 10)
    print(sc.to_dict())

    for t in range(timesteps):
        m = Market()
        # TODO for loop over actors, which place bids and asks
        m.accept_bid((0, 0, 2, 0.2))
        m.accept_ask((0, 1, 2, 0.2))

        m.print()
        m.clear()
        print(m.get_all_matches())
