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
Simply is an agent-based market simulation tool that assigns market actors to clusters and can
perform different :ref:`matching_algorithms` with their bids and asks with some of the algorithms
taking the various clusters into account.

Run Simply
==========

Run a market simulation using a config file (For possible configuations see `simply/config.py` or example config file `examples/*.txt`.):

.. code:: bash

    ./main.py config.cfg


Or use the market wrapper to communicate order information directly to the market matching functions in a json format:

.. code:: bash

    ./market_wrapper.py orders.json


License
=======

