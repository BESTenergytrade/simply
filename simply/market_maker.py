import warnings
from typing import TYPE_CHECKING, Sized

import numpy as np
import pandas as pd

import simply.config as cfg
from simply.actor import Order
from simply.market import MARKET_MAKER_THRESHOLD, ASK, BID

if TYPE_CHECKING:
    from simply.scenario import Environment

MARKETMAKERID = "MarketMaker"


class MarketMaker:
    """ The MarketMaker represents the market maker and implements methods accordingly. This
    includes generating buy and sell orders according to a provided price time series with
    excessively large energy amounts so all orders meeting the price criteria can be matched.
    In contrast to an ordinary actor the MarketMaker is not located in a cluster, does not have a
    schedule, is not restricted by a battery capacity or a strategy. Instead the market maker acts
    as almost infinite source and sink of energy
    """

    def __init__(self, buy_prices: Sized, environment: 'Environment' = None,
                 sell_prices: np.array = None, buy_to_sell_function=None, **kwargs):
        self.environment = environment
        self.id = MARKETMAKERID
        self.cluster = kwargs.get("cluster", None)
        # All prices the market maker is paying to buy energy. Mostly the prediction of these
        # values is used and provided via property
        self.all_buy_prices = np.array(buy_prices)
        self.all_sell_prices = self.generate_sell_prices(buy_to_sell_function, sell_prices)

        # Since resetting of objects should be possible the initial values are stored. In case
        # of a reset they are triggered and overwrite the above possibly mutated prices
        self._buy_prices = buy_prices
        self._sell_prices = sell_prices
        self._buy_to_sell_function = buy_to_sell_function

        # Check if all buy prices are lower than the sell prices.
        self.check_prices()

        self.horizon = cfg.config.horizon
        self.pred = pd.DataFrame()
        self.traded = {}
        self.energy_sold = [0]
        self.energy_bought = [0]
        if environment is not None:
            # Overwriting the environment market_maker with this market_maker
            self.environment.add_actor_to_scenario(self)
            self.create_prediction()
        else:
            warnings.warn("MarketMaker was not added to environment. To use the MarketMaker in a "
                          "scenario add it through scenario.add_participant()")

    def reset(self):
        # Reset prices to original prices
        self.all_buy_prices = np.array(self._buy_prices)
        self.all_sell_prices = self.generate_sell_prices(
            self._buy_to_sell_function, self._sell_prices)
        self.check_prices()
        self.create_prediction()
        self.traded = {}
        self.energy_sold = [0]
        self.energy_bought = [0]

    def check_prices(self):
        if not all(self.all_buy_prices <= self.all_sell_prices):
            warnings.warn(
                f"Not all buy prices are lower than the sell prices for the MarketMaker {self}")
            raise AssertionError

    def generate_sell_prices(self, buy_to_sell_function, sell_prices):
        if sell_prices is not None:
            if buy_to_sell_function is not None:
                warnings.warn("The market maker uses the provided selling prices. A function to "
                              "create selling prices from buying prices was provided as well, but "
                              "will not be used.")
            # if sell_prices are provided they are used
            return sell_prices
        else:
            if buy_to_sell_function is None:
                # if no specific function to calculate sell_prices is provided
                # the grid fee will be added to all the buy_price
                warnings.warn("Market Maker selling prices are set equal to buying prices")
                return self.all_buy_prices.copy()
            else:
                # if a specific function is provided, it is used to calculate sell_prices from
                # the buy_prices
                return buy_to_sell_function(self.all_buy_prices)

    def save_csv(self, dirpath):
        """
        Saves data and pred dataframes to given directory with actor specific csv file.

        :param str dirpath: (optional) Path of the directory in which the actor csv file should be
            stored.
        """
        save_data = ["all_buy_prices", "all_sell_prices"]
        save_df = pd.DataFrame()
        for data in save_data:
            save_df[data] = self.__dict__[data]
        save_df.to_csv(dirpath.joinpath(self.id + ".csv"))

    def to_dict(self, external_data=None):
        """
        Builds dictionary for saving.

        """
        return {
            "id": MARKETMAKERID,
            "sell_prices": list(self.all_sell_prices),
            "buy_prices": list(self.all_buy_prices)
            }

    def get_t_step(self):
        return self.environment.time_step
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

    def get_sell_prices(self):
        return self.pred.sell_prices
    # creating a property object
    sell_prices = property(get_sell_prices)

    def get_buy_prices(self):
        return self.pred.buy_prices
    # creating a property object
    buy_prices = property(get_buy_prices)

    def create_prediction(self):
        """Adjust predicted prices to the current time step"""
        self.pred["buy_prices"] = self.all_buy_prices[self.t_step: self.t_step + self.horizon]
        self.pred["sell_prices"] = self.all_sell_prices[self.t_step: self.t_step + self.horizon]

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
        mm_sell_order = Order(
            ASK, self.t_step, self.id, self.cluster, energy, self.current_sell_price)
        mm_buy_order = Order(
            BID, self.t_step, self.id, self.cluster, energy, self.current_buy_price)
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
            self.energy_sold[-1] += energy
        elif sign == +1:
            self.energy_bought[-1] += energy
        else:
            raise ValueError

    def prepare_next_time_step(self):
        self.energy_sold.append(0)
        self.energy_bought.append(0)
