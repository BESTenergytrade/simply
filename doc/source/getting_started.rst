~~~~~~~~~~~~~~~
Getting started
~~~~~~~~~~~~~~~

Simply is an open-source tool to simulate electricity markets using different matching algorithms.
..
   ToDo: expand on the description of simply

Documentation
=============

The latest documentation can be found `here <https://simply.readthedocs.io/en/latest/>`_.

Installation
============

The simply package is still in development and has not been officially released on `PyPi <https://pypi.org/>`_. This
means it is not yet listed and indexed in the PyPi repository for general distribution.

In order to use simply,  the `simply GitHub repository <https://github.com/BESTenergytrade/simply>`_ is cloned for local usage and development.
After cloning the repository, a virtual environment is created (e.g. using virtualenv or conda):

 .. code:: bash

    git clone git@github.com:BESTenergytrade/simply.git
    # create virtual environment
    virtualenv venv --python=python3.8
    source venv/bin/activate

Then there are two options to use simply:

#.
 Use the cloned repository directly and install the necessary dependencies using:

 .. code:: bash

    pip install -r requirements.txt

#.
 Or install the package in editable mode using:

 .. code:: bash

    pip install -e .

The tool uses Python (>= 3.8) standard libraries as well as specific, but well known libraries
such as `matplotlib <https://matplotlib.org/>`_, `pandas <https://pandas.pydata.org/>`_ and `networkx <https://networkx.org/>`_.


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
..
   ToDo: maybe adapt description a bit

Using Simply
============
The core of simply is the market matching algorithms. To be able to use the market matching algorithms, a
scenario is required as an input. The scenario can either be built from own data or randomly generated
by simply.

**Configuration file**

In both cases, a configuration file (`config.txt`) is required to specify the correct parameters
of the scenario and simulation. The file is split into the sections `scenario`, `market` and `outputs`, and
the parameters for each section are outlined as follows:

.. csv-table:: Scenario
   :file: "scenario_params.csv"
   :widths: 30, 70
   :header-rows: 1

Building your own scenario
--------------------------
..
   ToDo: check names of structure tree at the end and adapt if needed

A scenario is built from a number of required inputs: data (load, pricing, production, load directory), information on each
actor, information on the network and a configuration file. The structure to build a scenario should be set up
as follows:

::

    |-- projects
        |-- your_project_name
            |-- scenario_inputs
                |-- load
                    |-- your load timeseries
                |-- price
                    |-- your price timeseries
                |-- production
                    |-- your production timeseries
                |-- loads_dir.csv
            |-- actors_config.json
            |-- config.txt
            |-- network_config.json

**Scenario inputs**

The input timeseries data can be in either csv or json format. Below shows the generic format of the input timeseries.
The `Time` column contains entries for each interval in the format `YYYY-MM-DD hh:mm:ss`, where the interval time is
specified in `config.txt`. The number of entries must be equal to the number of timesteps
(also specified in `config.txt`). The second column contains the values for each interval for either load, production or
pricing, and `col_name` will change based on which data is represented.

::

    +---------------------+------------+
    |        Time         | col_name   |
    +=====================+============+
    | 2020-01-01 00:00:00 |    0.02    |
    +---------------------+------------+
    | 2020-01-01 00:00:15 |    0.05    |
    +---------------------+------------+
    |        ...          |    ...     |
    +---------------------+------------+

.. note:: There are no units set in simply, so all input files must be consistent with their units!

**Actors configuration**

The `actors_config.json` file represents a template for setting up a market community consisting of the market maker
and other market participants. For each market actor, the following must be specified, analogous to the example file:

#. The name of the market actor, e.g. "residential_1".
#. The market actor type, i.e. "market_maker", "residential", "industrial" or "business".
#. The location of the actor in the community network, i.e. the network node at which the prosumer is located.
#. The information about power consumption and power devices (if any):

    * The device type, i.e. "load", "solar" or "battery".
    * The device ID: here is the name of a file (.json or .csv), which is to be stored under /sample and contains the load curve for the respective power consumption or the respective power device.

Each actor is represented with the following structure:

::

  {
        "comment": "An example of a residential prosumer with load and pv data specifed by their 'deviceID'",
        "prosumerName": "residential_1",
        "prosumerType": "residential",
        "gridLocation": "N04",
        "devices": [
            {
                "deviceType": "load",
                "deviceID": "CHH10_sample.csv"
            },
            {
                "deviceType": "solar",
                "deviceID": "generated_pv.csv"
            }
        ]
    }

**Configuration file**

In the configuration file `config.txt` the correct settings regarding the scenario must be specified.

After setting up the inputs, you can run build_simulation



Generating a random scenario
----------------------------


Old text (to be deleted)
========================
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


After the network and the community have been created, `build_scenario.py` can be executed. The appropriate scenario
is created and saved to the location specified in the configuration file. The scenario contains a time series for each actor
with power generation, power consumption, and market demand or supply (including bid price). 




License
=======

