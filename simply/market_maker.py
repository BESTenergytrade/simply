import numpy as np
import pandas as pd
from typing import Iterable
import simply.config as cfg
from simply.actor import Order
from simply.market import MARKET_MAKER_THRESHOLD, ASK, BID

class MarketMaker:
    """ The MarketMaker represents the market maker and implements methods accordingly. This
    includes generating buy and sell orders according to a provided price time series with
    excessively large energy amounts so all orders meeting the price criteria can be matched.
    In contrast to an ordinary actor the MarketMaker is not located in a cluster, does not have a
    schedule, is not restricted by a battery capacity or a strategy. Instead the market maker acts
    as almost infinite source and sink of energy
    """

    def __init__(self,
                 scenario: 'simply.scenario.Scenario',
                 buy_prices: Iterable[float],
                 sell_prices: np.array = None,
                 buy_to_sell_function=None):
        self.scenario = scenario
        self.id = "MARKET_MAKER"
        # Price the market maker is paying to buy energy.
        self.buy_prices = np.array(buy_prices)
        if sell_prices is not None:
            # if sell_prices are provided they are used
            self.sell_prices = sell_prices
        else:
            if buy_to_sell_function is None:
                # if no specific function to calculate sell_prices is provided
                # the grid fee will be added to all the buy_price
                self.sell_prices = self.buy_prices.copy() + cfg.config.default_grid_fee
            else:
                # if a specific function is provided, it is used to calculate sell_prices from
                # the buy_prices
                self.sell_prices = buy_to_sell_function(self.buy_prices)
        assert all(self.buy_prices <= self.sell_prices)

        self.horizon = cfg.config.horizon
        self.pred = pd.DataFrame()
        self.create_prediction()
        self.traded = {}
        self.sold_energy = [0]
        self.bought_energy = [0]

        if self not in self.scenario.actors:
            self.scenario.actors.append(self)

    def get_t_step(self):
        return self.scenario.time_step
    # creating a property object
    t_step = property(get_t_step)

    def get_current_sell_price(self):
        return self.pred.sell_prices.iloc[0]
    # creating a property object
    current_sell_price = property(get_current_sell_price)

    def get_current_buy_price(self):
        return self.pred.buy_prices.iloc[0]
    # creating a property object
    current_buy_price = property(get_current_buy_price)


    def create_prediction(self):
        """Adjust predicted prices to the current time step"""
        self.pred["buy_prices"] = self.buy_prices[self.t_step: self.t_step + self.horizon]
        self.pred["sell_prices"] = self.sell_prices[self.t_step: self.t_step + self.horizon]

    def generate_orders(self):
        """
        Generate market maker orders for current time slot

        :return: generated new orders
        :rtype: list(Orders)
        """

        energy = MARKET_MAKER_THRESHOLD
        # ask  i.e. wanting to sell
        # bid  i.e. wanting to buy
        # Therefore the sign is the negative of the sign of the energy
        mm_sell_order = Order(ASK, self.t_step, self.id, None, energy, self.current_sell_price)
        mm_buy_order = Order(BID, self.t_step, self.id, None, energy, self.current_buy_price)
        orders = [mm_sell_order, mm_buy_order]
        return orders

    def receive_market_results(self, time, sign, energy, price):
        """
        Callback function when order is matched. Updates the actor's individual trading result.

        :param str time: time slot of market results
        :param int sign: for energy sold or bought represented by -1 or +1  respectively
        :param float energy: energy volume that was successfully traded
        :param float price: achieved clearing price for the stated energy
        """

        # append traded energy and price to actor's trades
        post = (sign * energy, price)
        pre = self.traded.get(time, ([], []))
        self.traded[time] = tuple(e + [post[i]] for i, e in enumerate(pre))
        if sign == -1:
            self.sold_energy[-1] += energy
        elif sign == +1:
            self.bought_energy[-1] += energy
        else:
            raise ValueError

    def next_time_step(self):
        self.sold_energy.append(0)
        self.bought_energy.append(0)
