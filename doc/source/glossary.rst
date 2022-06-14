Glossary
========
Scenario
--------
Reality or future to be regarded within the simulations. Especially defining what parameters will be considered or have an impact on other relevant parameters, e.g.:

- Prosumer

  - actor type (industry/household)
  - community composition
  - asset composition (e.g. Consumer, PV, PV+Bat, PV+EV)

    - Peak power
    - Power dynamic and volatility
    - Energy demand
    - Flexibility (e.g. Battery, EV-availability)
- Policy regarding network charges
- Policy regarding change of supplier
- Digitization
- Grid expansion

Model
-----
The simulation framework models how individual actors, situated within a electricity network, can trade on a periodic market using specific matching algorithms.

Setups
------
A setup defines a specific scenario variant given the simulation framework's model.
A setup consists of a parameter configuration, which might include a data set.

- Parameter set:

  - Includes all parameters that define decisions about the specific scenario variant.
  - These might include changing components of the BEST energy trading system. E.g. regarding:

    - Market mechanism
    - Network charges model
    - Conditions for the commercial energy supplier

- Data set:

  - Includes all time series and other reference data files that are used in the simulation of a scenario variant.

Schedule
--------
Time series of energy quantities that is planed to be exchanged with the grid.

- It should take all energy assets at the prosumer site, flexibilities and future market prices into account.
- It currently is based on perfect-foresight but should reflect prediction uncertainty in the future
- Bids and asks are only placed for each next market time slot, but a forecast make sense in order to anticipate prices while managing asset flexibility (e.g. a battery)

