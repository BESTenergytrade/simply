# Energies project data

The energy data used for these scenarios stems from the eGon Project: https://ego-n.org/
- the raw data can be downloaded from the Open Energy Platform (OEP): https://openenergyplatform.org/
- these scenarios represent prosumers in adjacent low voltage regions (in simply modeled as `cluster`) within medium voltage level `mvgd_32866` (i.e. `lvgd_1009200004`, `lvgd_1143000002`, `lvgd_1153600002`, `lvgd_1166200002`) for which households were sub-sampled
- household characteristic and time series data in csv-format was prepared for the generation of a simply-scenario using the proposed `actors_config.json` and `scenario_inputs` folder

**Build scenarios (creating the simply scenario folders):**

```
python build_scenario.py projects/Energies/2035_noChange_strat4 --data_dir projects/Energies/scenario_inputs
python build_scenario.py projects/Energies/2035_dynPrice_strat4 --data_dir projects/Energies/scenario_inputs
python build_scenario.py projects/Energies/2035_AU_strat4_pricing0-002 --data_dir projects/Energies/scenario_inputs
```

- **Remark:** As stardard workflow we recommend to include the scenario_inputs folder into the individual project folders in order to ensure reproducibility in case the scenario_inputs at given dataset at `--data_dir` path is expected to change. If `--data_dir` argument is not set, it defaults to the project folder with subfolder `scenario_inputs`.

**Run scenarios (simulates the simply scenarios):**

```
python match_market.py projects/Energies/2035_noChange_strat4
python match_market.py projects/Energies/2035_dynPrice_strat4
python match_market.py projects/Energies/2035_AU_strat4_pricing0-002
```
