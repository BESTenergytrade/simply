#!/usr/bin/env python3

from argparse import ArgumentParser

from simply import scenario, market, market_2pac, market_fair
from simply.config import Config
from simply.util import summerize_actor_trading


if __name__ == "__main__":
    parser = ArgumentParser(description='Entry point for market simulation')
    parser.add_argument('config', nargs='?', default="", help='configuration file')
    args = parser.parse_args()

    cfg = Config(args.config)
    if cfg.path.exists() and not cfg.update_scenario:
        sc = scenario.load(cfg.path)
    else:
        sc = scenario.create_random(12, 11)
        sc.save(cfg.path)
    # TODO make output folder for config file and Scenario json files, output series in csv and plots files

    if cfg.show_plots:
        sc.power_network.plot()
        sc.actors[0].plot(["load", "pv"])

    # Fast forward to interesting start interval for PV energy trading
    for a in sc.actors:
        a.t = cfg.start

    # generate requested market
    if "pac" in cfg.market_type:
        m = market_2pac.TwoSidedPayAsClear(0)
    elif cfg.market_type in ["fair", "merit"]:
        m = market_fair.BestMarket(0, sc.power_network)
    else:
        # default
        m = market.Market(0)

    for t in cfg.list_ts:
        m.t = t
        for a in sc.actors:
            # TODO concurrent bidding of actors
            order = a.generate_order()
            m.accept_order(order, a.receive_market_results)

        m.clear(reset=cfg.reset_market)
        if cfg.show_prints:
            print("Matches of bid/ask ids: {}".format(m.matches))
            print(
                "\nCheck individual traded energy blocks (splitted) and price at market level"
            )

    if cfg.show_prints:
        print("\nTraded energy volume and price at actor level")
        print(summerize_actor_trading(sc))
