.. _scenarios:

~~~~~~~~~~~~~~~~~~~
Scenarios
~~~~~~~~~~~~~~~~~~~

When running simply as a stand-alone simulation tool the `match_market.py` script
can be used to load, generate and save a scenario and run it for the configured
parameters (e.g. the number of simulations steps or :ref:`matching_algorithms`).

.. _config:

**Configuration File**
----------------------

In all cases for using simply, a configuration file (`config.cfg`) is required to specify the correct parameters
of the scenario and simulation. If a parameter is not specified in `config.cfg` and there is a default option,
this will be chosen. The file is split into the sections `scenario`, `market` and `outputs`, and
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
and other market participants. For each market actor, the following must be specified, analogous to the example file:

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

Under **"nodes"** the names of the individual nodes are listed (e.g. N01, N02). The market maker as
a special market participant does not have to be represented.

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
