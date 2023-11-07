run simply
==========
# Getting Started
**setup**
```sh
# create virtual environment
virtualenv venv --python=python3.8
source venv/bin/activate
# (1) install dependencies
pip install -r requirements.txt
# or (2) use setup
pip install -e .
```

**test**
```sh
# install testing dependencies
pip install -r tests/requirements.txt
# run tests
python -m pytest
```

**run**
```sh
python match_market.py config.txt
```

To run the example scenario, use the config file, `examples/config_example.txt`. Results can be 
checked in the `example_scenario` folder.

If `config.txt` or individual fields are not specified, default values are used as described in 
`simply/config.py`.

**gsy-e wrapper**
In order to use the simply matching algrithms in gsy-e through the [myco API](https://github.com/gridsingularity/gsy-myco-sdk), the wrapper can be used to translate the json string of gsy-e orders to the simply market model as shown in `market_wrapper.py`.

Example run, while loading input from json file `orders.json` in gsy-e format:
```sh
python market_wrapper.py orders.json
```
Documentation
=============
- [Documentation Index](https://simply.readthedocs.io/en/latest/)
