~~~~~~~~~~~~~~~
Getting started
~~~~~~~~~~~~~~~

**Simply is an open-source tool to simulate electricity markets using different matching algorithms.**

.. contents::
    :depth: 2
    :local:
    :backlinks: top

Introduction
===============
Simply is an electricity market simulation frame work consisting of scripts for

* scenario generation,
* market simulation and
* results visualisation and analysis.

Simply is an agent-based market simulation tool with market actors sending bids and asks to a
market, that can be cleared using different periodic `Matching Algorithms <https://simply.readthedocs.io/en/latest/matching_strategies.html>`_.
The algorithms take grid fees into account, which can be based on clusters defined by the agent's
location in the network.
The matching algorithms can also be used individually via a wrapper using a json format for order
definition.

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

1. Use the cloned repository directly and install the necessary dependencies using:

 .. code:: bash

    pip install -r requirements.txt

2. Or install the package in editable mode using:

 .. code:: bash

    pip install -e .

The tool uses Python (>= 3.8) standard libraries as well as specific, but well known libraries
such as `matplotlib <https://matplotlib.org/>`_, `pandas <https://pandas.pydata.org/>`_ and `networkx <https://networkx.org/>`_.


Using Simply
============
The core of simply is the market :ref:`matching_algorithms`. To be able to use the market matching algorithms, a
:ref:`scenarios` is required as an input. The scenario can either be built from own data or randomly generated
and simulated by simply as described below.

In all cases for using simply, a :ref:`config` (`config.cfg`) is required to specify the correct parameters
of the scenario and simulation.

.. _run_build_scenario:

Create your own scenario
------------------------

In order to build a scenario in a format that simply understands please prepare the data
as described by :ref:`build_scenario` (in :ref:`scenarios`), to define which data is associated
to which actor of the community and how they are connected by the network.

**Running build_scenario**

After the network and the community definitions have been set up, `build_scenario.py` can be executed. This is done by:

 .. code:: bash

    python build_scenario.py path/to/your/project/dir

with the option of specifying a path for your scenario inputs if you want to store them outside of your project directory:

 .. code:: bash

    python build_scenario.py path/to/your/project/dir -- data_dir path/to/your/scenario/inputs

The scenario is then created and automatically saved to `path/to/your/project/dir/scenario`. The scenario contains a
time series for each actor with power generation, power consumption, and market demand or supply (including bid price).

An example of a scenario can be found in `projects/example_projects/example_project`.

**Generating a randomized scenario**

There is also the option of generating a random scenario right before matching using `match_market.py`
(as explained in the section below). In this case, the parameters `nb_actors`,
`nb_nodes` and `weight_factor` should be specified in `config.cfg`, otherwise the default parameters are used. The only
input required before running the main simply function is the `config.cfg` file with attribute `load_scenario = False`:

::

    |-- projects
        |-- your_project_name
            |-- config.cfg

An example of config file and a generated random scenario can be found in `projects/example_projects/random_scenario`.
For more details please also see :ref:`scenarios`.

.. _run_simulation:

Running the simulation
----------------------

A simulation is executed by running the `match_market.py` script:

 .. code:: bash

    python match_market.py path/to/your/project/dir

If you choose to generate a random scenario, the scenario folder will be created in path/to/your/project/dir
scenario - here is where time series for each actor with power generation, power consumption, and market demand or supply
(including bid price) can be found.

For both instances, once you run `match_market.py` the results will be stored in path/to/your/project/dir/market_results.
Here you can see the results in csv-format for the matches and orders of actors in the network
as well as individual results per actor.

License
=======

MIT License

Copyright (c) 2021 BEST project developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
