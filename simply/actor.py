import random
import warnings

import numpy as np
import pandas as pd
from collections import namedtuple
import matplotlib.pyplot as plt
from pytest import approx

from simply.battery import Battery
from simply.util import daily, gaussian_pv
import simply.config as cfg

Order = namedtuple("Order", ("type", "time", "actor_id", "cluster", "energy", "price"))
Order.__doc__ = """
Struct to hold order

:param type: sign of order, representing bid (-1) or ask (+1)
:param time: timestamp when order was created
:param actor_id: ID of ordering actor
:param energy: amount of energy the actor wants to trade. Will be rounded down(asks)/up(bids)
    according to the market's energy unit
:param price: bidding/asking price for one unit of energy
"""


class Actor:
    """
    Actor is the representation of a prosumer, i.e. is holding resources (load, photovoltaic/PV)
    and defining an energy management schedule, generating bids or asks and receiving trading
    results.
    The actor interacts with the market at every time step in a way defined by the actor strategy.
    The actor fulfills his schedule needs by trading energy. A successful energy trade can
    be guaranteed by placing orders with at least the market maker price, since the market maker
    is seen as unlimited supply. At the start of every time step the actor can place one order to
    buy or sell energy at the current time step. Basis for this order are a predicted schedule and
    a market maker price time series as input.
    After matching took place, the (monetary) bank and resulting soc is calculated taking into
    consideration the schedule and the acquired energy of this time step, i.e. bank and soc at the
    end of the time step. Afterward the time step is increased and a new prediction for the
    schedule and price is generated.

    :param int id: unique identifier of the actor
    :param pandas.DataFrame() df: DataFrame, column names "load", "pv" and "price" are processed
    :param .battery.Battery() battery: Battery used by the actor
    :param str csv: Filename in which this actor's data should be stored
    :param float ls: (optional) Scaling factor for load time series
    :param float ps: (optional) Scaling factor for photovoltaic time series
    :param dict pm: (optional) Prediction multiplier used to manipulate prediction time series based
        on the data time series
    :param int cluster: cluster in which actor is located
    :param int strategy: Number for strategy [0-3]
    :param .scenario.Environment() environment: Environment reference for the actor

    Members:

    id : str
        Identifier of the actor to be set on creation
    grid_id : str
        [unused] Location of the actor in the network (init default: None)
    t : int
        Actor's current time slot should equal current market time slot (init default: 0)
    horizon : int
        Horizon up to which energy management is considered
        (default defined in simply.config)
    load_scale : float
        Scaling factor for load time series (default: init ls)
    pv_scale : float
        Scaling factor for photovoltaic time series (default: init ps)
    error_scale : float
        [unused] Noise scaling factor (default: 0)
    battery : object
        Representation of a battery (default: None)
    data : pandas.DataFrame()
        Actual generation and load time series as would be measured (default: init df)
    pred : pandas.DataFrame()
        Assumption of generation and load time series as would be predicted
        (default: init df + error)
    csv_file : str
        Filename in which this actor's data should be stored
    self.orders : list
        List of generated orders
    self.traded : dict
        Dictionary of received trading results per time slot including matched energy and clearing
        prices
    self.cluster: int
        cluster in which actor is located
    self.strategy: int
        Number for strategy [0-3]
    self.environment: .scenario.Environment()
        environment reference for the actor
    self.pricing_strategy: object
        Strategy used to calculate prices for orders, which are planned in the future.
        Type can be function(steps,price,energy) or dict with keys ("name", "params"). For further
        information check Pricing Strategies in ReadTheDocs or the documentation for
        :py:func:`get_price`
    self.battery: .battery.Battery()
        Battery used by the actor
    self.bank: float
        cumulated earnings and costs of this actor similar to a bank account balance
    self.matched_energy_current_step: float
        Amount of matched energy by orders of this actor in the current time step
    self.predicted_soc: np.array()
        predicted soc values for future time steps based on the battery state and planned actor
        behaviour
    self.market_schedule: np.array()
        planed energy amounts for interaction with the market_maker and basis of order generation

    """

    def __init__(self, id, df, environment=None, battery=None, csv=None, ls=1, ps=1, pm={},
                 cluster=None, strategy: int = 0, pricing_strategy=None):
        """
        Actor Constructor that defines an ID, and extracts resource time series from the given
         DataFrame scaled by respective factors as well as the schedule on which basis orders
         are generated.
        """
        self.id = id
        self.grid_id = None
        self.cluster = cluster

        self.horizon = cfg.config.horizon

        self.bank = 0
        self.matched_energy_current_step = 0
        self.socs = []
        self.predicted_soc = None
        self.load_scale = ls
        self.pv_scale = ps
        self.error_scale = 0
        self.battery = battery
        if self.battery is None:
            self.battery = Battery(capacity=2*cfg.config.energy_unit)
        self.data = pd.DataFrame()
        self.pred = pd.DataFrame()
        self.pm = pd.DataFrame()
        self.strategy = strategy
        self.pricing_strategy = pricing_strategy
        if csv is not None:
            self.csv_file = csv
        else:
            self.csv_file = f'actor_{id}.csv'
        # ToDo remove schedule from input or only allow either (load and pv) OR (schedule)
        for column, scale in [("load", ls), ("pv", ps), ("schedule", 1)]:
            self.data[column] = scale * df[column]
            try:
                self.pm[column] = np.array(pm[column])
            except KeyError:
                prediction_multiplier = self.error_scale * np.random.rand(self.horizon)
                self.pm[column] = prediction_multiplier.tolist()

        self._environment = None
        if environment is not None:
            # Let the actor have a reference to the environment and add it to the other
            # scenario actors. This is done through the environment property which also creates a
            # market schedule and prediction for the environment time step
            self.environment = environment
            environment.add_actor_to_scenario(self)

        self.orders = []
        self.traded = {}
        self.args = {"id": id, "df": df.to_json(), "csv": csv, "ls": ls, "ps": ps,
                     "pm": pm}

    def get_environment(self):
        return self._environment

    def set_environment(self, environment):
        self._environment = environment
        self.create_prediction()
        self.market_schedule = np.zeros(self.horizon)
        self.market_schedule[0] = self.get_default_market_schedule()[0]

    # creating a property object. This way changing environment also leads to updates
    environment = property(get_environment, set_environment)

    def get_mm_buy_prices(self):
        env = self.environment
        grid_fee = env.get_grid_fee(bid_cluster=env.market_maker.cluster, ask_cluster=self.cluster)
        # the achievable prices the mm buys energy for from the actor are reduced by the grid fee
        return env.market_maker.buy_prices-grid_fee
    # creating a property object
    mm_buy_prices = property(get_mm_buy_prices)

    def get_mm_sell_prices(self):
        env = self.environment
        grid_fee = env.get_grid_fee(ask_cluster=env.market_maker.cluster, bid_cluster=self.cluster)
        # the prices for which the mm sells energy to the actor are increased by the grid fee
        return self.environment.market_maker.sell_prices+grid_fee

    # creating a property object
    mm_sell_prices = property(get_mm_sell_prices)

    def get_t_step(self):
        return self.environment.time_step
    # creating a property object
    t_step = property(get_t_step)

    # getter
    def get_steps_per_hour(self):
        return self.environment.steps_per_hour
    steps_per_hour = property(get_steps_per_hour)

    def get_market_schedule(self, strategy=None):
        """ Generates a market_schedule for the actor which represents the strategy of the actor
        when to buy or sell energy. At the current time step the actor will always buy/ or sell
        this amount even at market maker price.

        :param strategy: Number representing the actor strategy from 0 to 3
        :type strategy: int
        :return: market_schedule with planed amounts of energy buying/selling per time step
        """
        possible_choices = [0, 1, 2, 3]
        if strategy is None:
            strategy = self.strategy
        if strategy not in possible_choices:
            warnings.warn(
                f"Strategy choice: {strategy} was not found in the list of possible "
                f"strategies: {possible_choices}. Using default strategy 0 without "
                "planning instead.")
            strategy = 0
        elif strategy != 0:
            if self.battery is None or self.battery.capacity == 0:
                warnings.warn(
                    f"Strategy choice: {self.strategy} was found but can not be used since "
                    f"the battery capacity is 0 or no battery exists. Using default strategy "
                    f"without planning instead.")
                strategy = 0

        if strategy == 0:
            self.market_schedule = self.get_default_market_schedule()
            # overwrite the current value of the market schedule if the soc would surpass 1
            self.predicted_soc = self.predict_socs()
            return self.market_schedule

        self.market_schedule = self.plan_global_self_supply()
        if strategy == 1:
            # overwrite the current value of the market schedule if the soc would surpass 1
            if self.predicted_soc[0] > 1:
                self.market_schedule[0] -= (self.predicted_soc[0] - 1) * self.battery.capacity
            return self.market_schedule

        self.market_schedule = self.plan_selling_strategy()
        if strategy == 2:
            return self.market_schedule
        self.market_schedule = self.plan_global_trading()
        return self.market_schedule

    def get_default_market_schedule(self):
        """ Return the default market schedule

        The default market schedule is the schedule with flipped signs and minor adjustments
        to make use of residual energy in the battery. An energy need with negative sign in the
        schedule is met with buying energy in the market_schedule which has a positive sign

        :return: default market schedule
        """
        default_market_schedule = -np.array(self.pred.schedule.values)
        # If there is energy in the battery, try making use of it in the next time step. This can
        # happen due to small differences in between traded energy and needed energy due to the
        # energy unit
        default_market_schedule[0] -= self.battery.energy()
        return default_market_schedule

    def predict_socs(self, clip=False, clip_value=1, planning_horizon=None):
        """ Returns prediction of future socs based on schedule and market schedule

        Creates a prediction for the planning horizon, which is not necessarily the horizon.
        The prediction is based on the soc of the battery, the schedule and the market schedule,
        which contains the amount the actor plans to buy or sell in a future time slot.
        Setting clip to true, does not allow values above the clipping value.

        :param planning_horizon: how far in the future should socs be predicted
        :param clip: should the values be clipped
        :param clip_value: value where socs are clipped to
        :return: soc prediction for planning horizon
        """
        if planning_horizon is None:
            planning_horizon = self.horizon - 1

        # cumulative scheduled energy per time step plus battery energy up to the planning horizon
        cum_energy_demand = (self.pred.schedule.cumsum() + self.market_schedule.cumsum() +
                             self.battery.soc * self.battery.capacity)[:planning_horizon+1]
        #  Effect the cumulative schedule would have on SOC
        soc_prediction = np.ones(planning_horizon+1) * self.battery.soc \
            + (cum_energy_demand - self.battery.soc * self.battery.capacity) / self.battery.capacity

        last_val = 0

        # find the last value, which is a proper value
        for counter in range(len(soc_prediction)-1, -1, -1):
            if not np.isnan(soc_prediction[counter]):
                last_val = soc_prediction[counter]
                break
        # this value is used to fill up nan array values
        soc_prediction[counter:] = last_val
        if clip:
            clip_soc(soc_prediction, clip_value)
        return soc_prediction

    def plan_global_self_supply(self):
        """Returns market_schedule where energy needs are covered by the lowest price slots.

        This strategy predicts the soc of the battery. If the soc would drop below 0 the
        market_schedule will cover the energy need by finding the cheapest time slot before or at
        the energy need time slot.
        :return: market_schedule
        """
        # initialize the market schedule with last market_schedule. Last value is 0 since
        # its a new value. This implies the planned strategy stays valid, i.e. a time slot for
        # optimal buying or selling does not change. This increases the speed of the algorithm
        # drastically
        self.market_schedule = np.roll(self.market_schedule, -1)
        self.market_schedule[-1] = 0
        soc_prediction = self.predict_socs(clip=True)
        planning_horizon = self.horizon - 1

        # Go through the cumulated demands, deducting the demand if we plan on buying energy
        for i, _ in enumerate(soc_prediction):
            energy = soc_prediction[i] * self.battery.capacity
            while energy < 0:
                # where is the lowest price in between now and when the energy is needed?
                # only check price where the battery is not full and time before when the energy
                # is needed
                possible_global_prices = np.ones(self.horizon) * float('inf')
                # prices are set where the soc in not full yet
                possible_global_prices[(soc_prediction < 1 - cfg.config.EPS)] = \
                    self.mm_sell_prices[(soc_prediction < 1 - cfg.config.EPS)]

                # index for the last inf value between now and energy demand
                last_inf_index = np.argwhere(possible_global_prices[:i + 1] >= float('inf'))
                if len(last_inf_index) == 0:
                    last_inf_index = 0
                else:
                    last_inf_index = last_inf_index.max()
                possible_global_prices[0:last_inf_index] = float('inf')
                # storing energy before that is not possible. only look at prices afterwards
                min_price_index = np.argmin(possible_global_prices[:i + 1])

                # cheapest price for the energy is when the energy is needed --> no storage needed
                if min_price_index == i or last_inf_index >= i:
                    self.market_schedule[i] -= energy
                    soc_prediction = self.predict_socs(clip=True, clip_value=1)
                    break

                # cheapest price is some time before the energy is needed. Check the storage
                # how much energy can be stored in the battery
                max_soc = min(1, max(0, np.max(soc_prediction[min_price_index:i])))
                max_storable_energy = (1 - max_soc) * self.battery.capacity

                # how much energy can be stored in the battery per time step via c-rate
                max_storable_energy = min(max_storable_energy, self.battery.capacity *
                                          self.battery.max_c_rate / self.steps_per_hour)

                # how much energy needs to be stored. Energy needs are negative
                stored_energy = min(max_storable_energy, -energy)
                # reduce the energy needs for the current time step
                energy += stored_energy

                # fix the soc prediction for the time span between buying and consumption
                # soc_prediction[min_price_index:i] += stored_energy / self.battery.capacity
                # not needed since soc prediction will be adjusted next iteration

                self.market_schedule[min_price_index] += stored_energy
                # Energy will be bought this time step. Predictions in the future, e.g. after this
                # time step will use the reduced demand for the time steps afterwards
                soc_prediction = self.predict_socs(clip=True, clip_value=1)

            # if the predicted soc is 1 for the time steps before the current one, it is not
            # possible to buy energy before. Since planning used only for time step 0, rest can
            # be skipped
            if max(soc_prediction[:i+1]) >= 1-cfg.config.EPS:
                # set planning horizon for soc_prediction
                planning_horizon = i
                break

        soc_prediction = self.predict_socs(clip=False, planning_horizon=planning_horizon)
        self.predicted_soc = soc_prediction

        return self.market_schedule

    def plan_selling_strategy(self):
        """Returns market_schedule where overcharges are sold at the highest possible price.

        This strategy extends strategy 1. If the soc would rise over 1 the
        market_schedule will sell the energy by finding the most expensive time slot before or at
        the energy need time slot. Soc constrains will be considered, meaning that if needed,
        multiple time slots will be used to sell the energy.
        :return: market_schedule
        """
        soc_prediction = self.predict_socs(clip=False)
        soc_prediction = soc_prediction[:self.horizon]

        # ToDo selling is not capped by max c-rate of battery
        # iterate over socs
        for i, _ in enumerate(soc_prediction):
            energy = soc_prediction[i] * self.battery.capacity
            overcharge = energy - self.battery.capacity
            # if overcharge is found, find the possible prices to sell this energy
            while overcharge > 0:
                possible_prices = self.mm_buy_prices.copy()[:self.horizon]
                # in between now and the peak, the right most/latest zero soc does not allow
                # reducing the soc before. Energy most be sold afterwards
                zero_soc_indices,  = np.where(soc_prediction[:i] < 0 + cfg.config.EPS)
                if zero_soc_indices.size > 0:
                    right_most_peak = np.max(zero_soc_indices)+1
                else:
                    right_most_peak = 0
                possible_prices[:right_most_peak] = float('-inf')
                highest_price_index = np.argmax(possible_prices[:i + 1])

                # best price is at the time of energy generation, therefore no restrictions towards
                # amounts of energy apply
                if highest_price_index == i:
                    self.market_schedule[i] -= overcharge
                    soc_prediction = self.predict_socs(clip=False)
                    break

                # if the energy is NOT sold at the time of production storage limitations need to be
                # taken into account

                # the time span in which energy could be sold to reduce the overcharge ends with
                # the overcharge itself and goes back for as long as no 0 soc is found. This is
                # the possible starting point for selling. In this time span the highest price was
                # found at the highest_price_index. At this index only as much energy can be sold as
                # the min predicted soc, in between this index and the time of overcharge, provides.
                soc_to_zero = np.min(soc_prediction[highest_price_index:i + 1])
                assert soc_to_zero <= 1 + cfg.config.EPS
                energy_to_zero = soc_to_zero * self.battery.capacity
                sellable_energy = min(energy_to_zero, overcharge)
                self.market_schedule[highest_price_index] -= sellable_energy
                overcharge -= sellable_energy
                soc_prediction = self.predict_socs(clip=False)

            # early breaking in case of reaching zero soc
            if min(soc_prediction[:i+1]) <= 0+cfg.config.EPS:
                break

        soc_prediction = self.predict_socs(clip=False)
        self.predicted_soc = soc_prediction
        return self.market_schedule

    def plan_global_trading(self):
        """ Strategy to buy energy when profit is predicted by selling the energy later on
               when the flexibility is given"""
        soc_prediction = self.predict_socs(clip=False)

        # planning to trade for the current time step is only possible until a soc of 1 is reached.
        planning_horizon = np.argwhere([soc_prediction >= 1-cfg.config.EPS])
        if planning_horizon.size == 0:
            planning_horizon = self.horizon - 1
        else:
            planning_horizon = planning_horizon[0, 1]-1
        soc_prediction = soc_prediction[:planning_horizon+1]
        # Note: Sell prices of the market maker are buy prices for the actor and vice versa
        buy_prices = np.array(self.mm_sell_prices)[:planning_horizon+1]
        sell_prices = np.array(self.mm_buy_prices)[:planning_horizon+1]
        # ToDo selling is not capped by max c-rate of battery
        # ++++++++++++++++++++
        sorted_buy_indexes = np.argsort(buy_prices)
        # go through all buying possibilities starting from the lowest price
        for buy_index in sorted_buy_indexes:
            buying_price = buy_prices[buy_index]
            possible_prices = sell_prices.copy()[:planning_horizon+1]

            # prices before buying can not be used for selling
            possible_prices[:buy_index + 1] = float("-inf")
            # if energy would be bought, it could be sold at these indices for a profit
            sell_indices = np.argwhere(possible_prices > buying_price)
            if sell_indices.size > 0:
                sell_indices = sell_indices.squeeze(axis=1)
            # if there are possible selling points of energy and there is the possibility of
            # storing energy in between, i.e soc<1
            while (sell_indices.size > 0 and
                    soc_prediction[buy_index:sell_indices[0]+1].max() < 1 - cfg.config.EPS):

                found_sell_index = None

                # make sure selling is not considered for sell_indices which lie behind an soc==1
                # event. They can not be used, since the battery can not be charged higher.
                socs = np.array(soc_prediction)
                soc_idx_almost_one = np.argwhere(socs[buy_index:] > 1 - cfg.config.EPS) + buy_index
                if soc_idx_almost_one.size > 0:
                    soc_idx_almost_one = soc_idx_almost_one.squeeze(axis=1)
                    left_most_soc_to_buy_index = min(soc_idx_almost_one)
                    # energy can not be stored over an SOC==1 event. Make selling price impossible
                    possible_prices[left_most_soc_to_buy_index:] = float("-inf")

                for sell_index in sell_indices:
                    sell_price = possible_prices[sell_index]
                    higher_sell_price_indices = np.argwhere(
                        possible_prices[sell_index + 1:] >= sell_price)
                    # highest price found
                    if len(higher_sell_price_indices) == 0:
                        found_sell_index = sell_index
                        break
                    else:
                        # there are higher prices. choose the left most higher price index
                        higher_sell_price_index = sell_index + 1 + higher_sell_price_indices.min()

                    lower_buy_price_indices = np.argwhere(
                        buy_prices[sell_index + 1:] <= sell_price)

                    # no buy dips but still higher selling prices
                    if len(lower_buy_price_indices) == 0:
                        continue
                    else:
                        # there are lower buy prices onward. choose the left most lower price index
                        lower_buy_price_index = sell_index + 1+lower_buy_price_indices.min()

                    # there are better selling points in the future
                    if higher_sell_price_index < lower_buy_price_index:
                        continue
                    # if not then we have the best selling point
                    else:
                        found_sell_index = sell_index
                        break
                # find how much energy can be stored in between buying and selling
                storable_energy = (
                    1-soc_prediction[buy_index:found_sell_index + 1].max()) * self.battery.capacity
                assert storable_energy > 0
                self.market_schedule[buy_index] += storable_energy
                self.market_schedule[found_sell_index] -= storable_energy
                soc_prediction = self.predict_socs(clip=False, planning_horizon=planning_horizon)
                assert 1 >= max(soc_prediction)-cfg.config.EPS
                assert 0 <= min(soc_prediction[:-1])+cfg.config.EPS
        self.predicted_soc = soc_prediction
        return self.market_schedule

    def plot(self, columns):
        """
        Plot columns from an actor's asset data and prediction with suffix label.

        :param str columns: name of the asset that should be plotted
        """
        pd.concat(
            [self.pred[columns].add_suffix("_pred"), self.data[columns]], axis=1
        ).plot()
        plt.show()

    def create_prediction(self):
        """Reset asset and schedule prediction horizon to the current planning time step self.t"""
        # ToDo add selling price
        for column in ["load", "pv", "schedule"]:
            if column in self.data.columns:
                self.pred[column] = (
                        self.data[column].iloc[self.t_step: self.t_step + self.horizon].
                        reset_index(drop=True) + self.pm[column])
        if "schedule" not in self.data.columns:
            self.pred["schedule"] = self.pred["pv"] - self.pred["load"]

    def update_battery(self, _cache=dict()):
        """Update the battery state with the current schedule and the matched energy in this step.

        This function needs to be called once per time step to track the energy inside of the
        battery. It takes the planned, i.e. predicted, schedule and changes the battery's SOC
        accordingly.

        :param _cache: cache of function calls, which SHOULD NOT be provided by user
        """
        # _cache keeps track of method calls by storing the last time of the method call at the
        # key of self/object reference. This makes sure that energy is only taken once per time step
        if self not in _cache:
            _cache[self] = self.t_step
        else:
            error = "Actor used the battery twice in a single time step"
            assert _cache[self] < self.t_step, error
            _cache[self] = self.t_step

        # assumes schedule is positive when pv is produced, Assertion error useful during
        # development to be certain
        assert self.pred.schedule[0] == approx(self.pred.pv[0] - self.pred.load[0])
        # ToDo Make sure that the balance of schedule and bought energy does not charge
        # or discharge more power than the max c rate
        self.battery.charge(self.pred.schedule[0] + self.matched_energy_current_step)

    def generate_orders(self):
        """
        Generate new orders for current time slot according to predicted schedule
        and both store and return it.

        :return: generated new orders
        :rtype: list(Order)
        """

        energy = self.market_schedule[0]
        assert not np.isnan(energy), "Market schedule is not a number. Is the simulation time " \
                                     "longer than the provided data?"
        if energy == 0 and self.pricing_strategy is None or all(self.market_schedule == 0):
            return []

        # the market schedule does not demand to buy or sell energy at the current time slot, but
        # a pricing strategy is provided which allows to generate orders for future demands, with
        # better than guaranteed prices
        if cfg.config.energy_unit > energy > -cfg.config.energy_unit:
            next_order_index = None
            for i, energy in enumerate(self.market_schedule):
                if energy != 0:
                    next_order_index = i
                    break
        else:
            next_order_index = 0

        if energy > 0:
            final_price = self.mm_sell_prices[next_order_index]
        else:
            final_price = self.mm_buy_prices[next_order_index]

        # Make sure not to over charge the battery since market schedule calculated the amount
        # for a later time slot
        energy = self.get_limited_energy(energy, index=next_order_index)
        energy = self.adjust_energy(energy, index=next_order_index)

        # get the price by using the pricing strategy
        price = self.get_price(next_order_index, final_price, energy)

        # TODO simulate strategy: manipulation, etc.
        # TODO take flexibility into account to generate the bid
        # TODO replace order type by enum
        # +1 as sign --> ask  i.e. wanting to sell
        # -1 as sign --> bid  i.e. wanting to buy
        # Therefore the sign is the negative of the sign of the energy
        new = Order(np.sign(-energy), self.t_step, self.id, self.cluster, abs(energy), price)
        self.orders.append(new)
        return [new]

    def get_price(self, steps, final_price, energy):
        """ Return the price for order generation for a planned future order generation.

        Planned order generation can be moved to an earlier time if energy storage is possible.
        Since urgency for energy procurement is lower, prices can be adjusted favorably for the
        actor. This method returns the current price based on the actor's attribute
        :py:attr:`~simply.actor.pricing_strategy` and a planned future order.

        :param steps: number of time steps until a future order will be placed
        :type steps: int
        :param final_price: price which would be used for a future order
        :type final_price: float
        :param energy: energy amount of future order. Positive energy stands for buying of energy
            , i.e. a lower price will be generated.
        :type energy: float
        :return: price for order generation at the current time step
        :rtype: float
        """
        return get_price(self.pricing_strategy, steps, final_price, energy)

    def get_limited_energy(self, energy, index):
        """ Return the amount of energy that can be ordered at the current time step

        Planned order generation can be moved forward if energy storage is possible. Energy storage
        can limit the amount of energy that can be bought/sold. This method returns the energy that
        can be procured at the current time step up to index. The limitation is calculated up to a
        future point in time. The index is the amount of simulation time steps up to this future
        point.

        :param energy: energy amount of future order. Positive energy stands for buying of energy
        :type energy: float
        :param index: number of time steps until a future order
        :type index: int
        :return: energy for order generation at the current time step
        :rtype: float
        """
        # no limit is needed if the energy is used in the current time step --> index 0
        if index == 0:
            return energy

        # look at all the time steps before the energy is traded with the market maker
        socs = self.predict_socs(planning_horizon=index-1)

        # buying energy
        if energy > 0:
            # The energy to be bought is bound by the remaining energy until fully charged battery
            # within the predicted horizon
            delta_soc = 1-socs.max()
            return max(min(energy, delta_soc*self.battery.capacity), 0)
        # selling energy
        else:
            # the energy to be sold is bound by the minimal energy level
            # within the predicted horizon
            delta_soc = -socs.min()
            # energy and delta_soc are negative valued
            return min(max(energy, delta_soc*self.battery.capacity), 0)

    def adjust_energy(self, energy, index=0):
        """ Adjust energy amount by up to one energy unit to stay in soc boundaries

        :param energy: energy amount for next order. Positive energy stands for buying of energy
        :type energy: float
        :param index: number of time steps until a future order
        :type index: int
        :return: energy amount for order generation the current time step
        :rtype: float
        """
        index = max(1, index)
        # buying energy
        if energy > 0:
            # rounding to the next energy unit can lead to unfulfilled schedules or below 0 socs.
            # In these cases increase the order by one energy unit, i.e. buy more energy
            if (self.battery.energy() + self.pred.schedule[0:index].sum() +
                    ((energy + cfg.config.EPS) // cfg.config.energy_unit *
                     cfg.config.energy_unit) < 0):
                energy += cfg.config.energy_unit

        # selling energy
        elif energy < 0:
            # rounding to the next energy unit can lead to unfulfilled schedules or over 1 socs.
            # In these cases decrease the order by one energy unit, i.e. sell more energy
            if self.battery.energy() + self.pred.schedule[0:index].sum() + (
                    ((energy+cfg.config.EPS) // cfg.config.energy_unit+1) *
                    cfg.config.energy_unit) > self.battery.capacity:
                energy -= cfg.config.energy_unit
        return energy

    def prepare_next_time_step(self):
        """Update actor and schedule and for next time step.

        Should be executed after clearing.
        Changes actor attributes according to the events in the current time step
        The events can be the impact of schedule and trading on the battery soc or the bank / cost
        for the actor in this time step.
        """

        if self.battery and not self.pred.empty:
            self.update_battery()
            self.socs.append(self.battery.soc)
        self.matched_energy_current_step = 0

    def receive_market_results(self, time, sign, energy, price):
        """
        Callback function when order is matched. Updates the actor's individual trading result.

        :param str time: time slot of market results
        :param int sign: for energy sold or bought represented by -1 or +1  respectively
        :param float energy: energy volume that was successfully traded
        :param float price: achieved clearing price for the stated energy
        """

        # order time and actor time have to be in sync
        assert time == self.t_step
        # sign can only take two values
        assert sign in [-1, 1]
        # append traded energy and price to actor's trades
        post = (sign * energy, price)
        pre = self.traded.get(time, ([], []))
        self.traded[time] = tuple(e + [post[i]] for i, e in enumerate(pre))
        # Buying energy as well as pv have positive sign
        self.matched_energy_current_step += energy*sign
        # Buying energy therefore decreases the bank
        self.bank += energy*(-sign)*price

        # if there is no market planning, e.g. market_schedule, no adjustment has to take place
        if self.market_schedule is None:
            return

        # if there is a market_schedule the matched energy has to be taken into account
        # iterate over the market schedule to reduce the market schedule according to
        # received energy
        delta_energy = sign*energy
        i = -1
        while np.sign(delta_energy) == sign and delta_energy != 0:
            i += 1
            if i == len(self.market_schedule):
                # energy amount of match was not found inside of the market schedule. Testing,
                # unexpected behaviour or self defined orders might be the reason. In this case give
                # warning and do not adjust market_schedule
                warnings.warn("Matched energy does not match planned energy.")
                return
            planned_energy = self.market_schedule[i]
            if planned_energy == 0:
                continue

            # make sure that the order that got placed and matched follows the sign of the
            # next energy need. Ignore small energies since difference occur due to energy unit
            if (not np.sign(delta_energy) == np.sign(planned_energy)
                    and abs(planned_energy) > 2 * cfg.config.energy_unit):
                warnings.warn("Matched energy does not match planned energy.")
            self.market_schedule[i] -= sign*min(abs(delta_energy), abs(planned_energy))
            delta_energy -= planned_energy

    def to_dict(self, external_data=False):
        """
        Builds dictionary for saving. external_data returns simple data instead of
        member dump.

        :param dict external_data: (optional) Dictionary with additional data e.g. on prediction
            error time series
        """
        args = self.args.copy()
        args["csv"] = self.csv_file
        if external_data:
            args_no_df = args
            args_no_df.update({"df": {}, "pm": {}, "ls": 1, "ps": 1})
            return args_no_df
        else:
            # since data is already scaled by ls and ps, both of these values are set to 1, so
            # they don't get applied twice
            args_df = args
            args_df.update({"df": self.data.to_json(), "pm": {}, "ls": 1, "ps": 1})
            return args_df

    def save_csv(self, dirpath):
        """
        Saves data and pred dataframes to given directory with actor specific csv file.

        :param str dirpath: (optional) Path of the directory in which the actor csv file should be
            stored.
        """
        # TODO if "predicted" values do not equal actual time series values,
        #  also errors need to be saved.
        if self.error_scale != 0:
            raise Exception('Prediction Error is not yet implemented!')
        save_df = self.data[["load", "pv", "schedule"]]
        save_df.to_csv(dirpath.joinpath(self.csv_file))


def create_random(actor_id, start_date="2021-01-01", nb_ts=24, ts_hour=1):
    """
    Create actor instance with random asset time series and random scaling factors

    :param str actor_id: unique actor identifier
    :param str start_date: Start date "YYYY-MM-DD" of the DataFrameIndex for the generated actor's
        asset time series
    :param int nb_ts: number of time slots that should be generated
    :param ts_hour: number of time slots per hour, e.g. 4 results in 15min time slots
    :return: generated Actor object
    :rtype: Actor
    """
    # Round to next full day
    extra_time_steps = -nb_ts % 24
    time_idx = pd.date_range(start_date, freq="{}min".format(int(60 / ts_hour)),
                             periods=nb_ts+extra_time_steps)
    cols = ["load", "pv", "schedule", "price"]
    values = np.random.rand(len(time_idx), len(cols))
    df = pd.DataFrame(values, columns=cols, index=time_idx)

    # Multiply random generation signal with gaussian/PV-like characteristic
    for day in daily(df, 24 * ts_hour):
        day["pv"] *= gaussian_pv(ts_hour, 3)

    # delete the added time steps
    if extra_time_steps > 0:
        df = df.drop(df.iloc[-extra_time_steps:].index)

    # Random scale factor generation, load and price time series in boundaries
    ls = random.uniform(0.5, 1.3)
    ps = random.uniform(1, 7)
    # Probability of an actor to possess a PV, here 40%
    pv_prob = 0.4
    ps = random.choices([0, ps], [1 - pv_prob, pv_prob], k=1)
    df["schedule"] = ps * df["pv"] - ls * df["load"]
    max_price = 0.3
    df["price"] *= max_price
    # Adapt order price by a factor to compensate net pricing of ask orders
    # (i.e. positive energy) Bids however include network charges
    net_price_factor = 0.7
    df["price"] = df.apply(lambda slot: slot["price"] - (slot["schedule"] > 0)
                           * net_price_factor * slot["price"], axis=1)
    # makes sure that the battery capacity is big enough, even if no useful trading takes place
    bat_capacity = max(random.random()*10, 2 * cfg.config.energy_unit)
    return Actor(actor_id, df, battery=Battery(capacity=bat_capacity), ls=ls, ps=ps)


def create_from_csv(actor_id, asset_dict={}, start_date="2021-01-01", nb_ts=None, ts_hour=1,
                    override_scaling=False):
    """
    Create actor instance with random asset time series and random scaling factors. Replace

    :param str actor_id: unique actor identifier
    :param Dict asset_dict: nested dictionary specifying 'csv' filename and column ('col_index')
        per asset or the time series index ('index') of the Actor
    :param str start_date: Start date "YYYY-MM-DD" of the DataFrameIndex for the generated actor's
        asset time series
    :param int nb_ts: number of time slots that should be generated, derived from csv if None
    :param ts_hour: number of time slots per hour, e.g. 4 results in 15min time slots
    :param override_scaling: if True the predefined scaling factors are overridden by the peak value
        of each csv file
    :return: generated Actor object
    :rtype: Actor
    """
    # Random scale factor generation, load and price time series in boundaries
    peak = {
        "load": random.uniform(0.8, 1.3) / ts_hour,
        "pv": random.uniform(1, 7) / ts_hour
    }
    # Probability of an actor to possess a PV, here 40%
    pv_prob = 0.4
    peak["pv"] = random.choices([0, peak["pv"]], [1 - pv_prob, pv_prob], k=1)

    # Initialize DataFrame
    cols = ["load", "pv", "schedule"]
    df = pd.DataFrame([], columns=cols)

    # Read csv files for each asset
    for col, csv_dict in asset_dict.items():
        # if csv_dict is empty
        if not csv_dict:
            continue
        csv_df = pd.read_csv(
            csv_dict["csv"],
            sep=',',
            parse_dates=['Time'],
            dayfirst=True
        )
        # Rename column and insert data based on dictionary
        df.loc[:, col] = csv_df.iloc[:nb_ts, csv_dict["col_index"]]
        # Override scaling factor by peak value (if True)
        if override_scaling:
            peak[col] = df[col].max()
        # Normalize time series
        df[col] = df[col] / df[col].max()
    # Set new index, in case it was not read in via asset_dict, it is generated
    if "index" not in df.columns:
        df["index"] = pd.date_range(
            start_date,
            freq="{}min".format(int(60 / ts_hour)),
            periods=nb_ts
        )
    df = df.set_index("index")

    # If pv asset key is present but dictionary does not contain a filename
    if "pv" in asset_dict.keys() and not asset_dict["pv"].get("filename"):
        # Initialize PV with random noise
        df["pv"] = np.random.rand(nb_ts, 1)
        # Multiply random generation signal with gaussian/PV-like characteristic per day
        for day in daily(df, 24 * ts_hour):
            day["pv"] *= gaussian_pv(ts_hour, 3)

    return Actor(actor_id, df, ls=peak["load"], ps=peak["pv"])


def clip_soc(soc_prediction, upper_clipping):
    """ Clip the soc values above the upper_clipping threshold.

    SOC predictions can be useful with and without soc restrictions. One restriction is that socs
    can not rise above a certain threshold, mostly 1. If this should be regarded, clipping of the
    socs is necessary. This is a more complex operation than just setting above 1 values to 1, since
    following values depend on the previous history, e.g. the socs history without clipping
    socs = [0.5, 1.0, 1.5, 1.8, 1.2, 1.3] in the clipped state is not [0.5, 1.0, 1.0, 1.0, 1.0, 1.0]
    but [0.5, 1.0, 1.0, 1.0, 0.4, 0.5]

    :param soc_prediction: soc_values which are clipped in place
    :type soc_prediction: np.array()
    :param upper_clipping: value above socs are clipped
    :type upper_clipping: float
    """
    soc_max = np.max(soc_prediction)
    while soc_max > upper_clipping:
        # descending array
        desc = np.arange(len(soc_prediction), 0, -1)
        # gradient of soc i.e. positive if charging negative if discharging
        diff = np.hstack((np.diff(soc_prediction), -1))
        # masking of socs >1 and negative gradient for local maximum
        # i.e. after lifting the soc, it finds the first spot where the soc is bigger
        # than the upper threshold and descending.
        idc_loc_max = np.argmax(desc * (soc_prediction > upper_clipping) * (diff < 0))

        # find the soc value of this local maximum
        soc_max = soc_prediction[idc_loc_max]
        # reducing everything after local maximum
        soc_prediction[idc_loc_max:] = soc_prediction[idc_loc_max:] - (soc_max - upper_clipping)

        # clipping everything before local maximum
        soc_prediction[:idc_loc_max][soc_prediction[:idc_loc_max] > upper_clipping] = upper_clipping
        soc_max = np.max(soc_prediction)


def get_price(pricing_strategy, steps, final_price, energy):
    """ Returns the price at the current time step to generate an order early

    :param pricing_strategy: strategy name and parameters or function with arguments steps, price
        and energy amount. Keys and values for the dictionary are dict("name":name,"param": param)
        with allowed names ["linear", "harmonic", "geometric"]. For documentation of the param
        values can be found at their respective functions, e.g.:

            - :py:func:`get_linear_price`
            - :py:func:`get_harmonic_price`
            - :py:func:`get_geometric_price`

        or in ReadTheDocs.

    :type pricing_strategy: dict() or function
    :param steps: amount of time step until interaction with market maker is planned
    :param final_price: price of the market maker interaction
    :param energy: amount of energy that is planned for trading. Positive amounts are buying and
        negative amounts are selling energy
    :return: current price for order generation
    :rtype: float
    """
    # the market schedule demands to buy or sell energy in the current time slot. Therefore
    # pricing will be adjusted to market maker prices, i.e. the final price will be used.
    if steps == 0:
        return final_price

    # a function can be given with the arguments steps, final_price and energy. it should return
    # a price. if even more advanced functions with access to the actor data shall be used the
    # get_price method can be over written
    if callable(pricing_strategy):
        return pricing_strategy(steps, final_price, energy)

    if not isinstance(pricing_strategy, type(dict())):
        raise TypeError("pricing_strategy is neither a callable function nor a dictionary.")

    # pricing strategy can be linear
    if pricing_strategy["name"] == "linear":
        return get_linear_price(steps, final_price, energy, pricing_strategy["param"])

    # pricing strategy can be harmonic
    if pricing_strategy["name"] == "harmonic":
        return get_harmonic_price(steps, final_price, energy, pricing_strategy["param"])

    # pricing strategy can be geometric
    if pricing_strategy["name"] == "geometric":
        return get_geometric_price(steps, final_price, energy, pricing_strategy["param"])

    raise ValueError("pricing_strategy is a dictionary but does not contain a valid strategy under "
                     "the key 'name'.")


def get_linear_price(steps, final_price, energy, param):
    """ Return the price based on a linear series

    :param steps: number of steps until the future order has to be met
    :type steps: int
    :param final_price: price for the future order (often equal to market maker price)
    :type final_price: float
    :param energy: amount of energy of the future order. Positive amounts mean buying energy
    :type energy: float
    :param param: gradient per time step, sign will be discarded
    :type param: list()
    :return:
    """
    try:
        m = param[0]
    except TypeError:
        m = param
    m = abs(m)
    sign = np.sign(energy)
    return final_price - sign * steps * m


def get_harmonic_price(steps, final_price, energy, param):
    """ Return the price based on a harmonic series

    Harmonic pricing changes the price according to the harmonic series meaning
    1, 1/2, 1/3, 1/4, 1/5 ... and so on.

    :param steps: number of steps until the future order has to be met
    :type steps: int
    :param final_price: price for the future order (often equal to market maker price)
    :type final_price: float
    :param energy: amount of energy of the future order. Positive amounts mean buying energy
    :type energy: float
    :param param: half_life_steps and symmetric_bound_factor, with symmetric_bound_factor being
        optional
    :type param: list()
    :return:
    """
    sign = np.sign(energy)
    # the half_life_steps is the steps where the price is 50% of the final price for buys.
    # if energy is supposed to be sold, its 200% instead
    # Note: This is not to be confused with an half-life time of an exponential decay function,
    # which is the behaviour of a geometric series. (see below)
    try:
        half_life_steps = param[0]
    except TypeError:
        half_life_steps = param
    error = "Harmonic series needs a positive non zero float value as first parameter"
    assert half_life_steps > 0, error
    factor = ((steps / half_life_steps) + 1) ** -1
    try:
        symmetric_bound_factor = param[1]
        if symmetric_bound_factor is None:
            raise IndexError
    except IndexError:
        # no symmetric_bound_factor was provided
        return factor ** sign * final_price

    # symmetric bound is always smaller than 1 and represents the convergence value of the
    # harmonic series. The default harmonic series convergences to 0 or diverges to infinity
    # if a symmetric_bound as second parameter is given the series will converge to
    # lim --> symmetric_bound_factor*final price instead
    # for sells it diverges to 1/symmetric_bound_factor*final price
    if symmetric_bound_factor > 1:
        symmetric_bound_factor = 1 / symmetric_bound_factor
    delta_price = (1 - symmetric_bound_factor) * final_price
    buy_price = (delta_price * factor + (final_price - delta_price))
    price = (buy_price / final_price) ** sign * final_price
    return price


def get_geometric_price(steps, final_price, energy, param):
    """ Return the price based on a geometric series

    Geometric pricing changes the price according to the geometric series,
    e.g. with the factor 0.5
    1, 1/2, 1/4, 1/8, 1/16 ... and so on.

    :param steps: number of steps until the future order has to be met
    :type steps: int
    :param final_price: price for the future order (often equal to market maker price)
    :type final_price: float
    :param energy: amount of energy of the future order. Positive amounts mean buying energy
    :type energy: float
    :param param: geometric_factor and symmetric_bound_cap, with symmetric bound capping being
        optional
    :type param: list()
    :return:
    """
    sign = np.sign(energy)
    # the geometric_factor is the factor with which the price is multiplied every time step
    try:
        geometric_factor = param[0]
    except TypeError:
        geometric_factor = param
    if geometric_factor > 1:
        geometric_factor = 1 / geometric_factor

    geometric_price = final_price * geometric_factor ** (steps * sign)
    try:
        symmetric_bound_cap = param[1]
        if symmetric_bound_cap is None:
            raise IndexError
    except IndexError:
        # no symmetric_bound_factor was provided
        return geometric_price
    if symmetric_bound_cap > 1:
        symmetric_bound_cap = 1 / symmetric_bound_cap

    if sign > 0:
        return max(geometric_price, symmetric_bound_cap * final_price)
    else:
        return min(geometric_price, 1 / symmetric_bound_cap * final_price)
