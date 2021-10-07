simply
======

```sh
virtualenv venv --python=python3.8
source venv/bin/activate
pip install -r requirements.txt

python main.py
```
simulaton set ups
==================
    # Actor time series
        ◦ load profile types: households, commerce/industry
        ◦ ressources: load, PV, battery (stationary, electric vehicle), CHP, wind-park
        ◦ Variation: scaling, (availability) and randomness 
    # format of data samples:
        ◦ folder structure: subfolder for different profile types
        ◦ column name = actor name
        ◦ load from .csv or .json
