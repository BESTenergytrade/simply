~~~~~~~~~~~~~~~
Getting started
~~~~~~~~~~~~~~~

Simply is a tool to simulate electricity markets using different matching algorithms.


Documentation
=============

Full documentation can be found `here <https://simply.readthedocs.io/en/latest/>`_

Installing Simply
=================

In order to use simply, first clone, then create virtual environments and install requirements as follows :

.. code:: bash

    git clone git@github.com:BESTenergytrade/simply.git
    # create virtual environment
    virtualenv venv --python=python3.8
    source venv/bin/activate
    # (1) install dependencies
    pip install -r requirements.txt

.. code:: bash

    # or (2) use setup
    pip install -e .

The tool uses Python (>= 3.8) standard libraries as well as specific, but well known libraries matplotlib, pandas and networkx.


General concept
===============
Simply is an electricity market simulation frame work consisting of scripts for 

* scenario generation, 
* market simulation and 
* results visualisation and analysis.

Simply is an agent-based market simulation tool with market actors sending bids and asks to a
market, that can be cleared using different periodic :ref:`matching_algorithms`.
The algorithms take grid fees into account, which can be based on clusters defined by the agent's
location in the network.
The matching algorithms can also be used individually via a wrapper using a json format for order
definition.

Run Simply
==========

Run a market simulation using a config file (For possible configuations see `simply/config.py` or example config file `examples/*.txt`.):

.. code:: bash

    ./main.py config.cfg


Or use the market wrapper to communicate order information directly to the market matching functions in a json format:

.. code:: bash

    ./market_wrapper.py orders.json


Create your own scenario 
========================

Under `config/` you will find two files (.json), which represent an example of how you can design your own scenario.  
The `example_config.json` file represents a template for setting up a market community consisting of the market maker 
and other market participants. For each market actor, the following must be specified, analogous to the example file: 

#. The name of the market actor, e.g. "residential_1".
#. The market actor type, i.e. "market_maker", "residential", "industrial" or "business". 
#. The location of the actor in the community network, i.e. the network node at which the prosumer is located. 
#. The information about power consumption and power devices (if any): 
    
* The device type, i.e. "load", "solar" or "battery". 
* The device ID: here is the name of a file (.json or .csv), which is to be stored under /sample and contains the load curve for the respective power consumption or the respective power device. 
 
The file `example_network.json` represents a template for the construction of a market community network. Under "nodes" 
the names of the individual nodes are listed (e.g. N01, N02). The market maker represents a separate node.  
Under "links" the network charge is defined for each combination of two nodes. Nodes between which there is a network 
charge of 0 represent a common cluster (see BEST Matching Algorithm). 

In the configuration file `config/config.txt` the correct settings regarding the scenario must be made. It can be set 
where the created scenario should be stored. The number of market actors must match the specifications in the `config.json` file. 
The number of nodes must match the information in `network.json`. 


After the network and the community have been created, `build_simulation.py` can be executed. The appropriate scenario 
is created and saved to the location specified in the configuration file. The scenario contains a time series for each actor
with power generation, power consumption, and market demand or supply (including bid price). 




License
=======

