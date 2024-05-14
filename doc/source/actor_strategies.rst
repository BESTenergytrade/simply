.. _actor_strategies:

~~~~~~~~~~~~~~~~~~~
Actor Strategies
~~~~~~~~~~~~~~~~~~~

Actor strategies are used to determine the timing of electricity trading, the amount of electrical energy to be traded,
and the order prices (buying and selling) for an actor. Each actor has constraints on when to buy or sell energy based on their
demand and if existing their supply of electrical energy, as well as their battery. The order price of the
actor corresponds to the price of the market maker at the time of electricity trading. In cases where, due to exceeding battery
constraints, energy has to be drawn from or fed into the grid, the actor sends an order with a price that corresponds to the
market maker prices (incl. grid fees). This way a successful trade is guaranteed.

There are four different actor strategies. Strategy 0 represents the simplest case, which is only based on the market maker's
electricity price of the current time slot. The other three strategies represent forecast-based,
market-oriented alternatives: The market maker's price time series as well as the actor's demand and supply of
electrical energy are considered for the near future given the configured horizon. Thus, the timing of
electricity trading based on future market maker prices permits the actor to
plan ahead in order to save costs or achieve higher profits through smart trading.

The extent to which matches are realized as a trade with the market maker depends on further orders from other actors given
the selected matching algorithm, applied grid fees and additional pricing strategies
(cf. :ref:`pricing_strategies` and :ref:`matching_algorithms`). As a result, this allows the actor - within its constraints - to
purchase energy at a lower price or sell energy at a higher price with respect to the guaranteed trading with the market maker.

The four actor strategies build on each other and are characterized by different features:

+--------------------------+--------------+--------------+--------------+--------------+
|                          | Strategy 0   | Strategy 1   | Strategy 2   | Strategy 3   |
+==========================+==============+==============+==============+==============+
| Forecast based purchase  |              | x            | x            | x            |
+--------------------------+--------------+--------------+--------------+--------------+
| Forecast based sale      |              |              | x            | x            |
+--------------------------+--------------+--------------+--------------+--------------+
| Time arbitrage           |              |              |              | x            |
+--------------------------+--------------+--------------+--------------+--------------+

From left to right, the strategies gain in economic advantage for the actor, but also in complexity. While an actor in
strategy 0 can only directly use or feed in energy from its own generation, in strategies 1 - 3 an actor can also have a
battery storage system, which enables it to temporarily store energy that is not used by the household or, for example,
an electric vehicle. In this way, the strategies serve self-consumption. By also selling electrical energy from the battery
in strategy 2 und 3 the economic efficiency of the generator and the battery storage system is increased.

Strategy 0
==========

Electrical energy is bought or sold at the moment it is needed or there is generation surplus. The energy is traded at
the price at which it is offered by the market maker. Since the future prices of the market maker are not considered and
a battery storage system is not used to improve profits from price fluctuations of the guaranteed market maker prices.

Strategy 1
==========

In strategy 1, the actor trades based on its own residual electricity demand, the state of charge (SOC) of the battery storage system, and
the prices of the market maker in the near future. This makes it possible to derive an ideal time to purchase
electricity that is ahead of actual demand and minimizes the cost of purchasing electricity. This is possible due to the
intermediate storage of energy in the battery. Electricity consumption and purchase can thus be decoupled in terms of
time. Excess energy from own generation is only fed into the grid when the battery storage system has reached a SOC of 1.

Strategy 2
==========

Strategy 2 uses strategy 1 and additionally considers the sale of electrical energy from own generation that has been
stored in the battery storage system. From the boundary conditions of the battery and the electricity generation and
demand, the times are derived at which the sale of electrical energy from the battery and / or current generation
achieves the highest price. Only those amounts of electrical energy are sold that would result in a SOC
above 1 and thus could not be stored. To do this, the algorithm compares the current SOC with the first SOC > 1 that
would result if all unused energy from own generation were stored in the battery. In case the amount of energy that
causes the positive deviation from the SOC of 1 can be sold the optimal time for selling is determined. If additional
energy can be sold it is checked if later time windows with a SOC > 1 can be served and if that would lead to maximum
profit.


Strategy 3
==========

Strategy 3 uses strategy 2 and considers time arbitrage to better exploit the market maker's dynamic prices.
If price fluctuations are strong enough to make it profitable to buy and later sell, the remaining capacity of the
battery is used to conduct this trade.



