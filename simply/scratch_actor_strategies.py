import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

# Building up a strategy for placing orders in the local market requires, rules to deduct a price
# from environmental conditions. This scratch file tries to implement a robust and sensible strategy,
# by building a semi-complex strategy from the bottom up.
# The first strategy will not take the local market into account, but just fill the battery by
# using the "global" energy market (glob_market)
# The second strategy will deduce a local_market price and place orders in accordance
# The third strategy will implement advanced strategies

# Boundary conditions, environment data and inputs
# For clarity, this paragraph is supposed to list all available data and boundary conditions for
# the actor and market
# glob_market has infinite energy available at every time step
# glob_market price is know at the current time step
# glob_market price is predicted for a prediction_horizon
# the prediction (pred.global_price) is a function of some random function/ blur and the true value, e.g. rand()*true_price
# the local market  has aggregated data (pred.schedule_aggr) about the schedule of all its participants, e.g. sum of all schedules
# schedules (pred.schedule) are a function of some random function/ blur and the true value, e.g. rand()*true_schedule
#   schedule= energy_produced-energy_needed, i.e. negative if energy is needed
# the strategy is implemented by each actor individually
# the strategy should "work" if all actors use the strategy
# the actor has a battery with the capacity of bat.capacity, with the same unit as energy schedule
# the battery has a maximum charging and discharging amount of N C, e.g. complete charge or discharge in
# 1/N hours. stored in battery.max_c_rate.
###############

## init of environment
prices = [0.2116079447, 0.1473127859, 0.22184087530000002, 0.11761082760000001, 0.2463169965,
          0.2020745841, 0.0113031114, 0.24701460990000002, 0.12690570210000002, 0.1467477666,
          0.0910571313, 0.1510937983, 0.0961995166, 0.16232426160000002, 0.1911430976, 0.2395885052,
          0.1161007245, 0.1912644558, 0.08394693780000001, 0.031559975000000004,
          0.07516904740000001, 0.0839614066, 0.1340712662, 0.1921131123]

schedule = [-0.2278688066, -0.4956801147, -0.5660800508, -0.4605807878, -0.7235523078,
            -0.41539310830000004, -0.0517064662, -0.4741886065, -0.253179973, -0.7055324580000001,
            -0.0665372924, -0.33647962400000003, -0.3992714075, -0.4354996278, -0.625752089,
            -0.30241824170000003, -0.23024240310000002, -0.6122942333, -0.1880810302, -0.1261036003,
            -0.18803270630000002, -0.2284269156, -0.7287319187, -0.0596583833]

schedule[7] += 2
schedule[15] -= schedule[15] * 4
schedule[19] -= schedule[19] * 4
schedule[20] -= schedule[20] * 3

const_grid_fee = 0.05
taxes_percent = 0.1


EPS = 1e-6
TIMEHORIZON = 24



class Battery:
    def __init__(self, capacity, max_c_rate, soc_initial=0.5):
        self.capacity = capacity
        self.max_c_rate = max_c_rate
        self.soc = soc_initial

    def energy(self):
        return self.soc * self.capacity

    def get_energy(self, energy):
        self.soc += energy / self.capacity

        # delta_soc_needed= energy/self.capacity
        # # Energy can only charge battery up to 1 and load can only discharge battery to 0
        # delta_soc_possible = max(min(delta_soc_needed,1-self.soc),-self.soc)
        # self.soc += delta_soc_possible
        # return delta_soc_possible*self.capacity


class Prediction:
    def __init__(self, prices, schedule):
        # use numpy for arrays for easier handling
        self.global_price = np.array(prices)
        self.schedule = np.array(schedule)
        self.selling_price = (np.array(prices) - const_grid_fee) * (1 - taxes_percent)
        # self.selling_price[self.selling_price < 0] = 0.01

    def next_timestep(self):
        # Roll data to act like a new timestep
        for data in [self.global_price, self.schedule, self.selling_price]:
            data[:] = np.roll(data, -1)


#########################################
# Strategy 1
# The first strategy will not take the local market into account, but just fill the battery by
# using the "global" energy market (glob_market). It will try to fulfill the schedule as cheaply
# as possible. It will only use a look up in the future as far as pred.schedule allows.
# There will not be energy buying or selling for needs or production behind this horizon.
#

class Actor:
    def __init__(self, prediction: Prediction, battery: Battery):
        self.global_sell_price = []
        self.pred = prediction
        self.battery = battery
        self.cost = 0
        self.global_market_buying_plan = np.array([])
        self.bought_energy_from_global_market = []
        self.bought_energy_from_local_market = []
        self.predicted_soc = np.array([])
        self.schedule = []
        self.global_price = []
        self.global_buy = []
        self.bought_energy = 0
        self.socs = []
        self.order = (0, 0, 0)

    def buy_planed_energy_from_global_market(self):
        bought_energy = self.global_market_buying_plan[0]
        self.schedule.append(self.pred.schedule[0])
        self.global_price.append(self.pred.global_price[0])
        self.bought_energy_from_global_market.append(bought_energy)
        self.battery.get_energy(self.pred.schedule[0] + bought_energy)
        self.socs.append(self.battery.soc)
        self.cost -= bought_energy * self.pred.global_price[0]

    def buy_order(self, local=False):
        order_amount, order_price, order_index = self.order
        self.battery.get_energy(order_amount)
        self.cost -= order_amount * order_price
        if not local:
            self.bought_energy_from_global_market.append(order_amount)
            self.bought_energy_from_local_market.append(0)
        else:
            self.bought_energy_from_global_market.append(0)
            self.bought_energy_from_local_market.append(order_amount)

    def write_data(self):
        self.schedule.append(self.pred.schedule[0])
        self.global_price.append(self.pred.global_price[0])
        self.global_sell_price.append(self.pred.selling_price[0])
        self.global_buy.append(self.global_market_buying_plan[0])
        self.battery.get_energy(self.pred.schedule[0])
        self.socs.append(self.battery.soc)

    def plan_selling_strategy(self):
        # Find peaks of SOCs above 1.
        #                                 x
        #             (x)               x   x
        #                   x         x      x
        #           x    x     x    x
        # 1 ----- x-------------  x-
        #      x         o
        #   x
        cum_energy_demand = self.pred.schedule.cumsum() + self.global_market_buying_plan.cumsum() + self.battery.soc * self.battery.capacity
        soc_prediction = np.ones(TIMEHORIZON) * self.battery.soc \
                         + (cum_energy_demand - self.battery.soc * self.battery.capacity) / \
                         self.battery.capacity
        for i, energy in enumerate(cum_energy_demand):
            overcharge = energy - self.battery.capacity
            while overcharge > 0:
                possible_prices = self.pred.selling_price.copy()
                possible_prices[soc_prediction < 0 + EPS] = float('-inf')
                # in between now and the peak, the right most/latest Zero soc does not allow
                # reducing the soc before. Energy most be sold afterwards
                zero_soc_indicies = np.where(soc_prediction[:i] < 0 + EPS)
                if np.any(zero_soc_indicies):
                    right_most_peak = np.max(zero_soc_indicies)
                else:
                    right_most_peak = 0
                possible_prices[:right_most_peak] = float('-inf')
                highest_price_index = np.argmax(possible_prices[:i + 1])

                # If the energy is sold at the time of production no storage is needed
                if highest_price_index == i:
                    self.global_market_buying_plan[i] -= overcharge
                    cum_energy_demand[i:] -= overcharge
                    break
                # current_soc_after_schedule=((self.battery.soc*self.battery.capacity)
                #                             +self.pred.schedule[0]+self.global_market_buying_plan[0])/self.battery.capacity
                soc_to_zero = min(np.min(soc_prediction[highest_price_index:i + 1]), 1)
                energy_to_zero = soc_to_zero * self.battery.capacity
                sellable_energy = min(energy_to_zero, overcharge)
                self.global_market_buying_plan[highest_price_index] -= sellable_energy
                cum_energy_demand[highest_price_index:] -= sellable_energy
                overcharge -= sellable_energy
                soc_prediction = np.ones(TIMEHORIZON) * self.battery.soc \
                                 + (cum_energy_demand - self.battery.soc * self.battery.capacity) \
                                 / self.battery.capacity

        soc_prediction = np.ones(TIMEHORIZON) * self.battery.soc \
                         + (cum_energy_demand - self.battery.soc * self.battery.capacity) \
                         / self.battery.capacity
        self.predicted_soc = soc_prediction

    def plan_global_self_supply(self):
        cum_energy = self.pred.schedule.cumsum() + self.battery.soc * self.battery.capacity
        self.global_market_buying_plan = np.array([0] * TIMEHORIZON).astype(float)
        # Go through the cumulated demands, deducting the demand if we plan on buying energy
        for i, energy in enumerate(cum_energy):
            while energy < 0:
                soc_prediction = np.ones(TIMEHORIZON) * self.battery.soc \
                                 + (
                                             cum_energy - self.battery.soc * self.battery.capacity) / self.battery.capacity
                # Where is the lowest price in between now and when I will need some energy
                # Only check prices where I dont expect a full soc already or
                # the time where the energy is needed
                possible_global_prices = np.ones(TIMEHORIZON) * float('inf')
                # prices are set where the soc in not full yet
                # possible_global_prices[(0- EPS<=soc_prediction )* (soc_prediction < 1 + EPS)] = \
                #     self.pred.global_price[(0 -EPS<soc_prediction )* (soc_prediction < 1 + EPS)]
                possible_global_prices[(soc_prediction < 1 - EPS)] = \
                    self.pred.global_price[(soc_prediction < 1 - EPS)]

                # index for the last inf value between now and energy demand
                last_inf_index = np.argwhere(possible_global_prices[:i + 1] >= float('inf'))
                if len(last_inf_index) == 0:
                    last_inf_index = 0
                else:
                    last_inf_index = last_inf_index.max()
                possible_global_prices[0:last_inf_index] = float('inf')
                # storing energy before that is not possible. only look at prices afterwards
                min_price_index = np.argmin(possible_global_prices[:i + 1])

                # cheapest price for the energy is when the energy is needed --> no storage is needed
                if min_price_index == i or last_inf_index >= i:
                    self.global_market_buying_plan[i] -= energy
                    cum_energy[i:] -= energy
                    break

                # cheapest price is some time before the energy is needed. Check the storage
                # how much energy can be stored in the battery
                max_soc = min(1, max(0, np.max(soc_prediction[min_price_index:i])))
                max_storable_energy = (1 - max_soc) * self.battery.capacity

                # how much energy can be stored in the battery per time step via c-rate
                max_storable_energy = min(max_storable_energy, self.battery.capacity
                                          * self.battery.max_c_rate / time_steps_per_hour)

                # how much energy do i need to store. Energy needs are negative
                stored_energy = min(max_storable_energy, -energy)
                # Reduce the energy needs for the current time step
                energy += stored_energy

                # fix the soc prediction for the time span between buying and consumption
                # soc_prediction[min_price_index:i] += stored_energy / self.battery.capacity
                self.global_market_buying_plan[min_price_index] += stored_energy
                # Energy will be bought this timestep. Predictions in the future, eg after this timestep
                # will use the reduced demand for the timesteps afterwards
                cum_energy[min_price_index:] += stored_energy

        soc_prediction = np.ones(TIMEHORIZON) * self.battery.soc \
                         + (
                                     cum_energy - self.battery.soc * self.battery.capacity) / self.battery.capacity
        self.predicted_soc = soc_prediction

    def plan_global_supply_(self):
        # cumulated energy for the time horizon
        # cum_energy_demand =[sum(self.pred.schedule[:i+1]) for i in range(TIMEHORIZON)]
        # optimal price for each timestep
        # plan of buying energy for the time horizon. Initialization with buying nothing
        self.global_market_buying_plan = np.array([0] * TIMEHORIZON).astype(float)

        # Soc in the future of current soc
        soc_prediction = np.array([self.battery.soc] * TIMEHORIZON)

        # plus the lifts through self production
        soc_lifts = np.zeros(TIMEHORIZON)
        soc_lifts[self.pred.schedule > 0] = self.pred.schedule[self.pred.schedule > 0]
        cum_energy_lift_through_production = np.cumsum(soc_lifts)

        for i, energy in enumerate(self.pred.schedule_after_buying):
            energy_before_prod = energy
            # energy needs are reduced by self production
            energy += min(cum_energy_lift_through_production[i], -energy)
            # cum energy lift is therefore reduced for further time steps
            cum_energy_lift_through_production[i:] += energy_before_prod - energy

            # Energy is needed, therefore we need to buy it in the time span from now to then.
            while energy < 0:
                # Where is the lowest price in between now and when I will need some energy
                # Only check prices where I dont expect a full soc already or
                # the time where the energy is needed

                possible_global_prices = np.ones(TIMEHORIZON) * float('inf')
                # prices are set where the soc in not full yet
                possible_global_prices[soc_prediction < 1 - EPS] = self.pred.global_price[
                    soc_prediction < 1 - EPS]
                # index for the last inf value between now and energy demand
                last_inf_index = np.argwhere(possible_global_prices[:i + 1] >= float('inf'))
                if len(last_inf_index) == 0:
                    last_inf_index = 0
                else:
                    last_inf_index = last_inf_index.max()
                possible_global_prices[0:last_inf_index] = float('inf')
                # storing energy before that is not possible. only look at prices afterwards
                min_price_index = np.argmin(possible_global_prices[:i + 1])

                # cheapest price for the energy is when the energy is needed --> no storage is needed
                if min_price_index == i or last_inf_index >= i:
                    bought_energy = -energy
                    energy += bought_energy
                    self.global_market_buying_plan[i] += bought_energy
                    break

                # cheapest price is some time before the energy is needed. Check the storage
                # how much energy can be stored in the battery
                max_storable_energy = (1 - np.max(
                    soc_prediction[min_price_index:i])) * self.battery.capacity

                # how much energy can be stored in the battery per time step via c-rate
                max_storable_energy = min(max_storable_energy, self.battery.capacity
                                          * self.battery.max_c_rate / time_steps_per_hour)

                # how much energy do i need to store. Energy needs are negative
                stored_energy = min(max_storable_energy, -energy)
                # Reduce the energy needs for the current time step
                energy += stored_energy

                # fix the soc prediction for the time span between buying and consumption
                soc_prediction[min_price_index:i] += stored_energy / self.battery.capacity
                self.global_market_buying_plan[min_price_index] += stored_energy
                # Energy will be bought this timestep. Predictions in the future, eg after this timestep
                # will use the reduced demand for this timestep
                if min_price_index == 0:
                    self.pred.schedule_after_buying[i] += stored_energy
        self.predicted_soc = soc_prediction

    # method for strategy 2
    def create_order(self):
        # Every actor has enough energy by implementing strategy 1. To make the local market attractive
        # each actor will want reduced cost per bought energy. For the current time step there is planned
        # buying of energy which is cost optimized by the global market. Since we want to guarantee
        # an improvement to strategy 1 we force the order to have lower prices than the market price

        ##                  1
        ## Index                2   3   4     5
        ## Planned Buying   1   0   0   0.5   0
        ## Schedule         0   0.5 0.5 0.25  0.25
        ## Price            1   3   4   1.5   2
        ## Predicted SOC    1   0.5 0   0.25  0

        ## Order Price      0.8         1.3
        ## Order Amount     1   0   0   0.5   0

        # at the current Index 1 strategy 1 will buy "1" energy since it is cheapest. Local market has to compete
        # with the global price by some kind of threshold say gm_threshold=0.8%
        # Global market Order is
        # order =Order(price = gm_threshold* pred.global_price[0], amount=global_market_buying_plan[0]))
        # the difference between the gotten local market energy and the needed energy is bought from
        # the global market
        ##                      2
        ## Index            1       3   4     5
        ## Planned Buying   1   0   0   0.6   0
        ## Schedule         0   0.5 0.5 0.3  0.3
        ## Price            1   3   4   1.5   2
        ## Predicted SOC    1   0.5 0   0.25  0
        ## Order Price      0.8         1.3
        ## Order Amount     1   0   0   0.5   0

        ## at the current index 2 strategy 1 wouldnt buy energy. The local market can compete with
        # the next buying already though. The next planned buying is at index 3 with 0.5 energy for
        # 1.5 price. Since there are 3 time steps before the energy is needed we offer some kind
        # of scaling function
        # order_amount = min(global_market_buying_plan[4], (1-Current_SOC)*capacity
        # order_price = scale_price(price=pred.global_price[4],timesteps=3)
        # order =Order(price = gm_threshold* pred.global_price[0], amount=order_amount))
        # for example
        gm_threshold = 0.9

        def scale_price(price, timesteps):
            if timesteps == 0:
                return price
            return price * gm_threshold ** timesteps

        next_global_buy = np.argwhere(self.global_market_buying_plan != 0)
        if len(next_global_buy) == 0:
            # better to raise Exception
            return 0, 0, 0

        next_global_buy = np.squeeze(next_global_buy).min()
        next_amount = self.global_market_buying_plan[next_global_buy]
        # ToDo
        # Buying prices below current global selling price as well as selling prices above current
        # global buying prices dont make sense since they will never be fullfilles in a proper
        # market. Implementation does not seem to be mandatory though
        if next_amount > 0:
            next_price = self.pred.global_price[next_global_buy]

            max_battery_soc = self.battery.soc + self.pred.schedule[0] / self.battery.capacity
            if next_global_buy != 0:
                max_battery_soc = max(max_battery_soc,
                                      np.max(self.predicted_soc[0:next_global_buy + 1]))
            order_amount = min(next_amount,
                               (1 - max_battery_soc) * self.battery.capacity)
            order_price = scale_price(next_price, next_global_buy)
        else:
            next_price = self.pred.selling_price[next_global_buy]

            min_battery_soc = self.battery.soc + self.pred.schedule[0] / self.battery.capacity
            if next_global_buy != 0:
                min_battery_soc = min(min_battery_soc,
                                      np.min(self.predicted_soc[0:next_global_buy + 1]))
            # todo higher orders than bat max charge/discharge could be allowed, since
            # production could go directly into the network
            order_amount = max(next_amount,
                               -min_battery_soc * self.battery.capacity)
            order_price = scale_price(next_price, -next_global_buy)
        return order_amount, order_price, next_global_buy

    def plan_global_trading_supply(self):
        """ Strategy to buy energy when profit is predicted by selling the energy later on
        when the flexibility is given"""
        cum_energy_demand = self.pred.schedule.cumsum() + self.global_market_buying_plan.cumsum() +\
                            self.battery.soc * self.battery.capacity
        soc_prediction = np.ones(TIMEHORIZON) * self.battery.soc \
                         + (cum_energy_demand - self.battery.soc * self.battery.capacity) / \
                         self.battery.capacity

        # +++++++++++++++++++++++++=
        # handle stored amount
        buy_index=0
        buying_price = 0
        sellable_amount_of_stored_energy=min(soc_prediction)*self.battery.capacity
        possible_prices = self.pred.selling_price.copy()
        possible_prices[:buy_index] = float("-inf")
        sell_indicies = np.argwhere(possible_prices > buying_price)
        if sell_indicies.size > 0:
            sell_indicies = sell_indicies.squeeze(axis=1)
        # If there are possible selling points of energy and there is the possibility of
        # storing energy in between, i.e soc<1
        while sellable_amount_of_stored_energy>0:

            found_sell_index = None
            for sell_index in sell_indicies:
                sell_price = possible_prices[sell_index]
                higher_sell_price_indicies = np.argwhere(
                    possible_prices[sell_index + 1:] >= sell_price)
                # highest price found
                if len(higher_sell_price_indicies) == 0:
                    found_sell_index = sell_index
                    break
                else:
                    # there are higher prices. choose the left most higher price index
                    higher_sell_price_index = higher_sell_price_indicies.min()
                lower_buy_price_indicies = np.argwhere(
                    self.pred.global_price[sell_index + 1:] <= sell_price)
                # No buy dips but still higher selling prices
                if len(lower_buy_price_indicies) == 0:
                    continue
                else:
                    # there are lower buy prices onward. choose the left most lower price index
                    lower_buy_price_index = lower_buy_price_indicies.min()
                # There are better selling points in the future
                if higher_sell_price_index < lower_buy_price_index:
                    continue
                # if not then we have the best selling point
                else:
                    found_sell_index = sell_index
                    break
            # find how much energy can be stored in between buying and selling
            storable_energy = max(
                soc_prediction[buy_index:found_sell_index + 1].max() * self.battery.capacity,
                self.battery.capacity)
            assert storable_energy > 0
            self.global_market_buying_plan[found_sell_index] -= sellable_amount_of_stored_energy
            cum_energy_demand[found_sell_index:] -= sellable_amount_of_stored_energy
            soc_prediction = np.ones(TIMEHORIZON) * self.battery.soc \
                             + (cum_energy_demand - self.battery.soc * self.battery.capacity) / \
                             self.battery.capacity
            sellable_amount_of_stored_energy=min(soc_prediction)*self.battery.capacity


        # ++++++++++++++++++++
        sorted_buy_indexes= np.argsort(self.pred.global_price)
        for buy_index in sorted_buy_indexes:
            buying_price = self.pred.global_price[buy_index]
            possible_prices = self.pred.selling_price.copy()
            possible_prices[:buy_index+1] = float("-inf")
            sell_indicies = np.argwhere(possible_prices>buying_price)
            if sell_indicies.size>0:
                sell_indicies=sell_indicies.squeeze(axis=1)
            # If there are possible selling points of energy and there is the possibility of
            # storing energy in between, i.e soc<1
            while sell_indicies.size>0 and soc_prediction[buy_index:sell_indicies[0]].max()<1-EPS:

                found_sell_index=None
                for sell_index in sell_indicies:
                    sell_price=possible_prices[sell_index]
                    higher_sell_price_indicies=np.argwhere(possible_prices[sell_index+1:]>=sell_price)
                    # highest price found
                    if len(higher_sell_price_indicies)==0:
                        found_sell_index=sell_index
                        break
                    else:
                        # there are higher prices. choose the left most higher price index
                        higher_sell_price_index= higher_sell_price_indicies.min()
                    lower_buy_price_indicies=np.argwhere(self.pred.global_price[sell_index+1:]<=sell_price)
                    # No buy dips but still higher selling prices
                    if len(lower_buy_price_indicies)==0:
                        continue
                    else:
                        # there are lower buy prices onward. choose the left most lower price index
                        lower_buy_price_index= higher_sell_price_indicies.min()
                    # There are better selling points in the future
                    if higher_sell_price_index<lower_buy_price_index:
                        continue
                    # if not then we have the best selling point
                    else:
                        found_sell_index = sell_index
                        break
                #find how much energy can be stored in between buying and selling
                storable_energy = max(soc_prediction[buy_index:found_sell_index+1].max()*self.battery.capacity, self.battery.capacity)
                assert storable_energy>0
                self.global_market_buying_plan[buy_index] +=storable_energy
                self.global_market_buying_plan[found_sell_index] -= storable_energy
                cum_energy_demand[buy_index:] +=storable_energy
                cum_energy_demand[found_sell_index:] -=storable_energy
                soc_prediction = np.ones(TIMEHORIZON) * self.battery.soc \
                                 + (cum_energy_demand - self.battery.soc * self.battery.capacity) / \
                                 self.battery.capacity

time_steps_per_hour = 1
prices = [p for price in prices for p in [price] * time_steps_per_hour]
schedule = np.array(
    [s for sched in schedule for s in [sched / time_steps_per_hour] * time_steps_per_hour])
schedule[5:10] += 2
schedule[18:22] += 2
schedule=np.array(schedule)*0
pred = Prediction(prices, schedule)
pred.next_timestep()
pred.next_timestep()

battery = Battery(capacity=3, max_c_rate=2, soc_initial=0.0)
actor = Actor(pred, battery)

for _ in range(30):
    actor.plan_global_self_supply()
    actor.plan_selling_strategy()
    actor.plan_global_trading_supply()
    actor.plan_selling_strategy()
    order_amount, order_price, order_index = actor.create_order()
    actor.order = (order_amount, order_price, order_index)
    if order_amount != 0:

        if (order_amount > 0 and
                (order_price >= actor.pred.global_price[0] - EPS) or
                (order_amount < 0) and
                (order_price <= actor.pred.selling_price[0] + EPS)):
            actor.buy_order(local=False)
        elif random.random() > 1.5:
            actor.buy_order(local=True)
        else:
            actor.bought_energy_from_local_market.append(0)
            actor.bought_energy_from_global_market.append(0)
    else:
        actor.bought_energy_from_local_market.append(0)
        actor.bought_energy_from_global_market.append(0)

    actor.write_data()
    pred.next_timestep()

f, ax = plt.subplots()
ax.plot(actor.bought_energy_from_global_market, label="Energy from GM")
ax.plot(actor.schedule, label="Schedule")
ax2= ax.twinx()
ax2.plot(actor.global_price,"k", label="Buy Price")
ax2.plot(actor.global_sell_price, linewidth=1, label="Sell Price")
ax.plot(actor.socs, label="SOC")
ax.plot(actor.bought_energy_from_local_market, label="Energy from LM")
ax.plot(actor.global_buy, "k--", linewidth=1, label="Planed Buy")

print("Actor 1 had costs of: ", actor.cost)
print("Actor 1 bought from GM", sum(actor.bought_energy_from_global_market))
print("Actor 1 bought from LM", sum(actor.bought_energy_from_local_market))
print("Actor 1 bought from TOTAL",
      sum(actor.bought_energy_from_global_market) + sum(actor.bought_energy_from_local_market))
print("~~~~~~~~~~~~~")
ax.legend()
ax2.legend()
plt.show()
a = 2



# optimal soc initial is 0 in this case, since we want a stationary battery level over the time-horizon
# i.e. power is bought for a already know timestep in the future, and there is no way of selling energy
# since we ignore the local market. Initial soc of 0 lets us store up to 1 battery capacity from the start
# Make a data point for each timestep
time_steps_per_hour = 1
prices = [p for price in prices for p in [price] * time_steps_per_hour]
schedule = [s for sched in schedule for s in [sched / time_steps_per_hour] * time_steps_per_hour]
pred = Prediction(prices, schedule)
pred.next_timestep()
pred.next_timestep()

battery = Battery(capacity=3, max_c_rate=2, soc_initial=0.0)
actor = Actor(pred, battery)

for _ in range(30):
    actor.plan_global_self_supply()
    actor.buy_planed_energy_from_global_market()
    pred.next_timestep()

f, ax = plt.subplots()
ax.plot(actor.bought_energy_from_global_market, label="Energy from GM")
ax.plot(actor.schedule, label="Schedule")
ax.plot(actor.global_price, label="Price")
ax.plot(actor.socs, label="SOC")
plt.legend()
print("Actor 1 had costs of: ", actor.cost)
print("Actor 1 bought from GM", sum(actor.bought_energy_from_global_market))
print("~~~~~~~~~~~~~")
a = 2

############
# Strategy 2
# The 2nd strategy will take the local market into account by trying to estimate useful order prices.
# It will not be greedy and use strategy as upper bound. Therefore it should not be able to have
# higher costs than strategy 1. This means it wont greedly hope for better prices after energy buys
# are planned by  strat. 1
time_steps_per_hour = 1
prices = [p for price in prices for p in [price] * time_steps_per_hour]
schedule = [s for sched in schedule for s in [sched / time_steps_per_hour] * time_steps_per_hour]
pred = Prediction(prices, schedule)
pred.next_timestep()
pred.next_timestep()

battery = Battery(capacity=3, max_c_rate=2, soc_initial=0.0)
actor = Actor(pred, battery)

for _ in range(30):
    actor.plan_global_self_supply()
    order_amount, order_price, order_index = actor.create_order()
    actor.order = (order_amount, order_price, order_index)
    if order_amount != 0:
        if order_price >= actor.pred.global_price[0] - EPS:
            assert order_index == 0
            actor.buy_order(local=False)
        elif random.random() > 0.5:
            actor.buy_order(local=True)
        else:
            actor.bought_energy_from_local_market.append(0)
            actor.bought_energy_from_global_market.append(0)
    else:
        actor.bought_energy_from_local_market.append(0)
        actor.bought_energy_from_global_market.append(0)

    actor.write_data()
    pred.next_timestep()

f, ax = plt.subplots()
ax.plot(actor.bought_energy_from_global_market, label="Energy from GM")
ax.plot(actor.schedule, label="Schedule")
ax.plot(actor.global_price, label="Price")
ax.plot(actor.socs, label="SOC")
ax.plot(actor.bought_energy_from_local_market, label="Energy from LM")
print("Actor 1 had costs of: ", actor.cost)
print("Actor 1 bought from GM", sum(actor.bought_energy_from_global_market))
print("Actor 1 bought from LM", sum(actor.bought_energy_from_local_market))
print("Actor 1 bought from TOTAL",
      sum(actor.bought_energy_from_global_market) + sum(actor.bought_energy_from_local_market))
print("~~~~~~~~~~~~~")
ax.legend()

a = 2

###########
# ToDo
# Battery class implementation with methods/functionality from above
# implementation of additional costs involved in global market price,
# eg. split buying and selling price (with taxes and grid fees)
################# Strategy 3
## Consider aggregated local market schedule
## Consider market price development
time_steps_per_hour = 1
prices = [p for price in prices for p in [price] * time_steps_per_hour]
schedule = np.array(
    [s for sched in schedule for s in [sched / time_steps_per_hour] * time_steps_per_hour])
schedule[5:10] += 2
schedule[18:22] += 2
pred = Prediction(prices, schedule)
pred.next_timestep()
pred.next_timestep()

battery = Battery(capacity=3, max_c_rate=2, soc_initial=0.0)
actor = Actor(pred, battery)

for _ in range(30):
    actor.plan_global_self_supply()
    actor.plan_selling_strategy()
    order_amount, order_price, order_index = actor.create_order()
    actor.order = (order_amount, order_price, order_index)
    if order_amount != 0:

        if (order_amount > 0 and
                (order_price >= actor.pred.global_price[0] - EPS) or
                (order_amount < 0) and
                (order_price <= actor.pred.selling_price[0] + EPS)):
            assert order_index == 0
            actor.buy_order(local=False)
        elif random.random() > 0.5:
            actor.buy_order(local=True)
        else:
            actor.bought_energy_from_local_market.append(0)
            actor.bought_energy_from_global_market.append(0)
    else:
        actor.bought_energy_from_local_market.append(0)
        actor.bought_energy_from_global_market.append(0)

    actor.write_data()
    pred.next_timestep()


f, ax = plt.subplots()
ax.plot(actor.bought_energy_from_global_market, label="Energy from GM")
ax.plot(actor.schedule, label="Schedule")
ax.plot(actor.global_price, label="Price")
ax.plot(actor.socs, label="SOC")
ax.plot(actor.bought_energy_from_local_market, label="Energy from LM")
ax.plot(actor.global_buy, "k--", linewidth=1, label="Planed Buy")
print("Actor 1 had costs of: ", actor.cost)
print("Actor 1 bought from GM", sum(actor.bought_energy_from_global_market))
print("Actor 1 bought from LM", sum(actor.bought_energy_from_local_market))
print("Actor 1 bought from TOTAL",
      sum(actor.bought_energy_from_global_market) + sum(actor.bought_energy_from_local_market))
print("~~~~~~~~~~~~~")
ax.legend()

plt.show()
a = 2

## Thoughts on further strategies.
## So far the strategies are all somewhat conservative. They could be made better by:
## Making orders not just according to the current next buy/sell but all the next sales, if enough
## flexibility is in the soc and the order prices are lower for both the next and the buy after that
## or the selling price is higher than the next optimal sell and the sell after that.
## The status of the local market is not taken into account as well.
## One easy way of taking the status of the local market into consideration would be an analysis
## of the summed up schedule of all the schedules. In simple terms . The more production there is,
## the greedier the price handling can be. Prices should still never exceed the global market
# boundary prices. Since there is no way of looking at production specificly the net of
# production and consumption has to be used. If we had some form of production curve, it would
# make sense to scale the price according to the sum of available production of energy in between
# the current timestep and the time where we buy theenergy from the global market.
# E.g. we use some random fitting value (fv=10) and the local market will produce 35 energy
# in between now, and when we buy the energy we could scale the price by some
# function like gm_price*(10/35)). As we get closer to the point of buying from the global market,
# price increases up the global market price.
# To achieve something similar we can use sum(arctan(net_energy/fv)+pi/2)
#
## make use of global market fluctuations. The strategy can be further improved by making use
## of global market fluctuations. Eg the buying price is cheaper than a selling price in the future.
#
#
#
#
#    ºBuying Price
#    •Selling Price                                         ○
#                                                        ○○
#                                                      ○
#                                              ○     ○      •• Highest Selling Price
#                                            ○  ○○○       •••
#                                           ○            •••
#                                          ○    ••   • ••
#         ○○○     ○○     ○○○           ○    ••• •••
#        ○    ○○○○  ○○○○    ○○○     ○○   •••
#      ○○    ○○                   ○ ○○    •••
#      ────────────────────────────────•─ ── ─ ── ─────────── Lowest buying price
#              ••       ••         •••
#           ••••••••••••••••    ••••
#      ••••• •             ••••••
## or more complicated. Here the optimal Strategy wouldnt be to buy at the start and sell everything
# at the end but to buy at the start. sell at the first high peak. buy at the low peak and sell
# again at the end
#                                         ○○○○○○○○○○
#                                        ○○            ○○
#                                        ○○  ••••••••••
#                                       ○   ••
#                                      ○  •••
#                                    ○○    ••
#          ○○○○○○○○○○            ○○ ••
#        ○○           ○○○       ○○   •••
#      ○○○     ••••    ○○○   ○○   •••
#    ○○     •••    •••   ○○○○○  •••
#  ○○    ••••        ••••••   •••
# ○○▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬ ▬▬▬▬▬ ▬▬▬▬▬▬▬▬▬▬▬▬▬  ▬  ▬▬ ▬
#    •••
#  •••
# Algorithm for buying energy to sell it later on. The selling strategy should take
# care of selling the energy at the lowest
# price already.
#
# For each buying price from now to the future where the predicted soc is below 1
#   Get selling Prices (SPH) which are higher
#   For each SPH check whats in the closer future. A higher SPH OR A lower buying price
#      if a lower buying price
#           found best point of selling with index (best_sell_index)
#           break
#   Get the possible amount you can buy for as 1-max(predicted_soc[buy_index:sell_index)
#   If the amount is bigger than 0 buy it--> predict the soc again

