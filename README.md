run simply
==========

```sh
virtualenv venv --python=python3.8
source venv/bin/activate
pip install -r requirements.txt

python main.py
```
glossary
========
## scenario
Imaginable reality or future with respect to parameters beyond our control. E.g.
- Prosumer composition)
  - power demand
  - energy demand
- Flexibilities
- Policy regarding Network charges
- Policy regarding change of supplier 
- Digitalisation
- Grid expansion

## variant
Changing components of the BEST energy trading system. E.g. regarding: 
 - Market mechanism
 - Network charges model 
 - Conditions for the commercial energy supplier

## model
Abstraction/simplification of a scenario (variant)
D3A together with a specific set up represents a model
   
## setups 
Setups generate a model of the scenario variant under consideration from the simulation framework.
A Setup consists of a parameter set and a data set.
### parameter set
Includes all parameters that define decisions about the scenario variant
### data set
Includes all time series and other data files that are used in the model

## schedule
Time series of energy quantities that should be traded.
It is generated taking all energy assets at the prosumer site, flexibilities and future market prices into account.
It is the output of obs-d/bidding agent.
It is not clear if such a thing will exist in final simulations since it might be that only bids and asks for the next market period will be 
calculated by OBS-d.
