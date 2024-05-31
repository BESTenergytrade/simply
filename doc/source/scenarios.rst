.. _scenarios:

~~~~~~~~~~~~~~~~~~~
Scenarios
~~~~~~~~~~~~~~~~~~~

When running simply as a stand-alone simulation tool the `match_market.py` script
can be used to load, generate and save a scenario and run it for the configured
parameters (e.g. the number of simulations steps or :ref:`matching_algorithms`).

A scenario consists of
- **market participants** (Actors and Market Maker) including individual data on which
orders are generated during a simulation
- **network** and **grid fee** definition regarding the location of Actors

.. _config:

Configuration File
------------------

In all cases for using simply, a configuration file (`config.cfg`) is required to specify the correct parameters
of the scenario and simulation. If a parameter is not specified in `config.cfg` and there is a default option,
this will be chosen.

Default Values
==============

The file is split into the sections `scenario`, `market` and `outputs`, and
the parameters for each section are outlined as follows:

.. csv-table:: Scenario
   :file: ../files_to_be_displayed/scenario_params.csv
   :widths: 30, 70, 30, 30
   :header-rows: 1

.. csv-table:: Market
   :file: ../files_to_be_displayed/market_params.csv
   :widths: 30, 70, 30, 30
   :header-rows: 1

.. csv-table:: Actor
   :file: ../files_to_be_displayed/actor_params.csv
   :widths: 30, 70, 30, 30
   :header-rows: 1

.. csv-table:: Output
   :file: ../files_to_be_displayed/output_params.csv
   :widths: 30, 70, 30, 30
   :header-rows: 1

.. _Building a Grid Fee Matrix from Configuration Files:

Building a Grid Fee Matrix from Config Files
============================================

**Composition from Config Files**

The grid fee matrix can be composed by reading entries from configuration files using the following attributes:

- ``default_grid_fee``: Default grid fee applied for trades with the market maker
- ``local_grid_fee``: Local grid fee applied for trades within a cluster
- ``weight_factor``: Factor describing the relationship of grid fee to cumulative power network edge weights (see description above)

**Example Configuration Entries**

.. code-block:: python

   #--------------------------
   # market
   #--------------------------
   # default grid_fee to be used by market maker
   default_grid_fee = 0.09
   # local grid fee to be used
   local_grid_fee = 0
   # factor describing the relation of grid fee to cumulative power network edge weights
   weight_factor = 0.03

The **weight_factor** describes the relationship between grid fee and cumulative power network edge weight
between two nodes in the network. The cumulative weight is the sum of all edge weights along the shortest
path in the network between those nodes. If there are multiple nodes per cluster, nodes of one cluster
have equivalent cumulative weights to all nodes of another cluster.

For the example project, this results in a grid fee matrix:

.. code-block:: python

    [[0, 0.03],
     [0.03, 0]]

The diagonal has a fee of 0 due to `local_grid_fee` parameter, while the multiplication of the
`weight_factor` and network weight of 1 between node N01 and N02 is resolved to a fee of 0.03.
The Market Maker, that is not part of a cluster, is attributed a `default_grid_fee` as configured.

.. _build_scenario:

Building your own scenario
--------------------------

A scenario is built using the `build_scenario.py` script (see :ref:`run_build_scenario`) from
a number of required inputs:

- data collection (`scenario_inputs`): load, pricing, generation, load timeseries
- information on each actor (`actors_config.json`)
- information on the network connecting the actors (`network_config.json`)
- a configuration file (`config.cfg`)

The structure to build a scenario can be set up as shown below.
Note that the directory containing your data timeseries (`scenario_inputs`) can be located elsewhere if you
specify in the command line (:ref:`run_build_scenario`). However, `actors_config`, `config` and `network_config` must all be stored in your project
directory:

::

    |-- projects
        |-- your_project_name
            |-- scenario_inputs
                |-- load
                    |-- your load timeseries
                |-- price
                    |-- your price timeseries
                |-- generation
                    |-- your generation timeseries
                |-- loads_dir.csv
            |-- actors_config.json
            |-- config.cfg
            |-- network_config.json

**Scenario inputs**

The input timeseries data can be in either csv or json format. Below shows the generic format of the input timeseries.
The `Time` column contains entries for each interval in the format `YYYY-MM-DD hh:mm:ss`, where the interval time is
specified in `config.cfg`. The number of entries must be greater than the specified number of simulation timesteps `nb_ts` plus the prediction `horizon`
(also specified in `config.cfg`). The second column contains the values for each interval for either load, generation or
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
(:class:`simply.market_maker.MarketMaker`) and other market participants.
For each market actor (:class:`simply.actor.Actor`), the following must be specified, analogous to the example file:

#. The name of the market actor, e.g. "residential_1".
#. The market actor type, i.e. "market_maker", "residential", "industrial" or "business".
#. The location of the actor in the community network, i.e. the network node at which the prosumer is located.
#. The information about power consumption and power devices (if any):

   - The device type, i.e. "load", "solar" or "battery".
   - The device ID: here is the name of a file (.json or .csv), which is to be stored under /sample and contains the load curve for the respective power consumption or the respective power device.

Each actor is represented with the following structure:

::

  {
        "comment": "An example of a residential prosumer with load and pv data specified by their 'deviceID'",
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


**Network configuration**

The file `network_config.json` represents a template for the construction of a market community network in a
common Graph output format with nodes and links.

Under **"nodes"** the names of the individual nodes are listed (e.g. N01, N02). The Market Maker as
a special market participant does not have to be represented in the network.

Under **"links"** the network charge is defined for each combination of two nodes. Nodes between which there is a network
charge of 0 represent a common cluster (see :ref:`best_matching`). The general structure is shown below:

::

    {
      "example_network": {
        "directed": false,
        "multigraph": false,
        "graph": {},
        "nodes": [
          {
            "id":  "N01"
          },
          {
            ... :  ...
          }
        ],
        "links": [
          {
            "weight": 0,
            "source": "N01",
            "target": "N02"
          },
          {
            ... : ...,
            ... : ...,
            ... : ...
          }
        ]
      }
    }



Finishing Simulation Setup
--------------------------

TODO

Methods to Enter Grid Fee Matrix
================================

Entering a grid fee matrix into a simulation is essential for accurate cost calculations related to using the grid.
Ensure proper format, identifiers, units, and values for the grid fee matrix, whether complete or created from config file entries.

There are two methods to define a grid fee matrix for the simply simulation:

1. :ref:`Providing a Complete Grid Fee Matrix`
2. :ref:`Building a Grid Fee Matrix from Configuration Files`

.. _Providing a Complete Grid Fee Matrix:

Providing an Already Complete Grid Fee Matrix
=============================================

**Format:**
The grid fee matrix should be provided in a structured format such as JSON.

**Placement:**
The grid fee matrix update events should be placed as specified in the configuration.

Requirements
============

**Names/Identifiers:**
Each row and column must be labeled with unique identifiers representing different clusters or nodes.

**Example Identifiers:**

- **Network Lines:** line_1, line_2, line_n
- **Nodes or Clusters:** node_A, node_B, node_C

**Units:**
Specify consistent units for both row and column headers to avoid confusion (e.g., USD/kWh, EUR/kWh).
Currently, fees are always attributed as a price rate (i.e. per kWh).
