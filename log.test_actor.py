============================= test session starts ==============================
platform linux -- Python 3.8.10, pytest-7.4.0, pluggy-1.2.0
rootdir: /home/friederike/Code/simply-stuff/simply/simply
collected 25 items

tests/test_actor.py .........................                            [100%]

=============================== warnings summary ===============================
simply/config.py:53
simply/config.py:53
  /home/friederike/Code/simply-stuff/simply/simply/simply/config.py:53: UserWarning: No Configuration file path was provided. Default values will be used.
    warnings.warn("No Configuration file path was provided. Default values will be used.")

simply/config.py:66
simply/config.py:66
  /home/friederike/Code/simply-stuff/simply/simply/simply/config.py:66: UserWarning: No project_dir was provided. Default project_dir ./projects/example_projects/example_project is used
    warnings.warn("No project_dir was provided. Default project_dir ./projects/example_projects/example_project is used")

tests/test_actor.py: 22 warnings
  /home/friederike/Code/simply-stuff/simply/simply/venv/lib/python3.8/site-packages/networkx/algorithms/shortest_paths/generic.py:137: DeprecationWarning: shortest_path for all_pairs will return an iterator in v3.3
    warnings.warn(msg, DeprecationWarning)

tests/test_actor.py: 1209 warnings
  /home/friederike/Code/simply-stuff/simply/simply/simply/market.py:261: UserWarning: At least one cluster is 'None', returning default grid fee.
    warnings.warn("At least one cluster is 'None', returning default grid fee.")

tests/test_actor.py::TestActor::test_recv_market_results
tests/test_actor.py::TestActor::test_rule_based_strategy_0
tests/test_actor.py::TestActor::test_rule_based_strategy_0
tests/test_actor.py::TestActor::test_rule_based_strategy_0
tests/test_actor.py::TestActor::test_rule_based_strategy_0
tests/test_actor.py::TestActor::test_rule_based_strategy_0
tests/test_actor.py::TestActor::test_rule_based_strategy_0
  /home/friederike/Code/simply-stuff/simply/simply/simply/actor.py:777: UserWarning: Matched energy does not match planned energy.
    warnings.warn("Matched energy does not match planned energy.")

tests/test_actor.py: 61 warnings
  /home/friederike/Code/simply-stuff/simply/simply/simply/actor.py:767: UserWarning: Matched energy does not match planned energy.
    warnings.warn("Matched energy does not match planned energy.")

tests/test_actor.py::TestActor::test_create_random_multidays_halfhour
  /home/friederike/Code/simply-stuff/simply/simply/simply/actor.py:839: SettingWithCopyWarning: 
  A value is trying to be set on a copy of a slice from a DataFrame.
  Try using .loc[row_indexer,col_indexer] = value instead
  
  See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    day["pv"] *= gaussian_pv(ts_hour, 3)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
====================== 25 passed, 1304 warnings in 6.10s =======================
