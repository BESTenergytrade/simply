.. _actor_strategies:

~~~~~~~~~~~~~~~~~~~
Actor Strategies
~~~~~~~~~~~~~~~~~~~

Actor strategies are used to determine the timing of electricity trading, the amount of electrical energy to be traded,
and the bid prices (buying and selling) for an actor. Each actor has constraints on when to buy or sell based on their
demand and possibly their supply of electrical energy, as well as their battery. The price bid of the
actor corresponds to the price of the market maker at the time of electricity trading. This ensures that electrical
energy can be drawn from the grid and, if necessary, that excess energy from own generation can be sold.

There are four different actor strategies. Strategy 0 represents the simplest case, where the actor only knows the
market maker's electricity price for the current time. The other three strategies represent forecast-based ,
market-oriented alternatives. The market maker's price time series as well as the respective actor's demand and supply of
electrical energy are now known for the near future. Thus, the timing of electricity trading with the market maker can
be planned in advance and costs can be saved for the actor or higher profits can be achieved through smart trading.

The extent to which the plan for trading electricity with the market maker is adhered to depends on whether further bids
from other actors exist and whether, as a result, energy can be purchased at a lower price or sold at a higher
price by the scheduled trading date at the latest (cf. :ref:`pricing_strategies` and :ref:`matching_algorithms`).

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
an electric car. In this way, the strategies serve self-consumption. By also selling electrical energy from the battery
in strategy 2 und 3 the economic efficiency of the power plant and the battery storage system is increased.

Strategy 0
==========

Electrical energy is bought or sold at the moment it is needed or there is generation surplus. The energy is traded at
the price at which it is offered by the market maker. Since the future prices of the market maker are not known and
there is no battery storage system, price fluctuations of the market maker cannot be used specifically.

Strategy 1
==========

In strategy 1, the actor knows its own electricity demand, the state of charge (SOC) of the battery storage system, and
the prices of the market maker in the near future. This makes it possible to derive an ideal time to purchase
electricity that is ahead of actual demand and minimizes the cost of purchasing electricity. This is possible due to the
intermediate storage of energy in the battery. Electricity consumption and purchase can thus be decoupled in terms of
time. Excess energy from own generation is only fed into the grid when the battery storage system has reached a SOC of 1.

Strategy 2
==========

Strategy 2 uses strategy 1 and additionally considers the sale of electrical energy from own generation that has been
stored in the battery storage system. From the boundary conditions of the battery and the electricity generation and
demand, the times are derived at which the sale of electrical energy from the battery and / or directly from the own
power plant achieves the highest price. Only those amounts of electrical energy are sold that would result in a SOC
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



