# REQUIREMENTS# pyomo package needs to be installed
# a solver needs to be installed: CBC or GLPK are open source; for CBC
# coppy cbc_solver (\\FS01\RL-Institut\05_Temp\Andreas\Pyomo\cbc_solver)
# in the directory where the repo clone is

import pyomo.environ as pyo
import pandas as pd


def optimize_schedule(df_actor, df_prices, capacity=10, max_c_rate=1, soc_initial=0.5):
    # parametrisation of battery specified in simply/battery.py
    # capacity=10, max_c_rate=1, soc_initial=0.5
    # TODO battery-efficiency ?

    # generate single time series vectors
    t = [0] * len(df_actor)
    load = [0] * len(df_actor)
    pv = [0] * len(df_actor)
    buy_prices = [0] * len(df_prices)
    sell_prices = [0] * len(df_prices)

    # fill time series vectors with data from df
    for i in range(len(df_actor)):
        t[i] = i
        load[i] = df_actor.iat[i, 1]
        pv[i] = df_actor.iat[i, 2]
        buy_prices[i] = df_prices.iat[i, 1]
        sell_prices[i] = df_prices.iat[i, 2]

    # other parameters
    # TODO minutes per time step can be obtained using datetime functions from input csv-files
    time_interval = 15  # minutes per time step
    # TODO grid fee for buying electricity, see config.cfg
    grid_fee = 0.0  # 0.09
    # TODO if no upper bound is provided the problem turns out to be unbounded
    grid_connection_capacity = 20

    # PYOMO OPTIMISATION MODEL
    model = pyo.ConcreteModel()
    model.charging_power = pyo.Var(t, bounds=(0, capacity * max_c_rate))
    model.discharging_power = pyo.Var(t, bounds=(0, capacity * max_c_rate))
    model.stored_energy = pyo.Var(t, bounds=(0, capacity))
    # 0/ 1 for distinguishing between charging / discharging
    model.bi_charge = pyo.Var(t, within=pyo.Binary)
    model.power_from_grid = pyo.Var(t, bounds=(0, grid_connection_capacity))
    model.power_to_grid = pyo.Var(t, bounds=(0, grid_connection_capacity))
    model.cash_flow = pyo.Var(t)

    # energy balance for the system
    model.energy_balance_system = pyo.ConstraintList()
    for i in range(len(t) - 1):
        model.energy_balance_system.add(
            0 ==
            model.power_from_grid[i] + pv[i] + model.discharging_power[i]
            - load[i] - model.power_to_grid[i] - model.charging_power[i])

    # component energy storage
    model.energy_balance_storage = pyo.ConstraintList()
    for i in range(len(t) - 1):
        # constraint
        model.energy_balance_storage.add(
            model.stored_energy[i+1] - model.stored_energy[i] ==
            (model.charging_power[i] - model.discharging_power[i]) * time_interval/60)

    model.start_storage = pyo.Constraint(
        expr=model.stored_energy[0] ==
        soc_initial * capacity)

    # optional: equal soc at first and last time step
    model.start_end_storage = pyo.Constraint(
        expr=model.stored_energy[0] == model.stored_energy[len(t)-1])

    # needed in order to not have a discharge that affects the timestep after the last considered
    model.end_no_discharge_storage = pyo.Constraint(
        expr=0 == model.discharging_power[len(t)-1])

    # binary variable to separate charging and discharging timesteps in order to
    # exclude having both at the same time
    model.binary_charge_storage = pyo.ConstraintList()
    for i in t:
        model.binary_charge_storage.add(
            model.charging_power[i] <= model.bi_charge[i] * capacity * max_c_rate)

    model.binary_discharge_storage = pyo.ConstraintList()
    for i in t:
        model.binary_discharge_storage.add(
            model.discharging_power[i] <= (1 - model.bi_charge[i]) * capacity * max_c_rate)

    # costs to be used in objective function
    model.cash_flow_equation = pyo.ConstraintList()
    for i in t:
        model.cash_flow_equation.add(
            model.cash_flow[i] ==
            sell_prices[i] * model.power_to_grid[i] * time_interval/60
            - (buy_prices[i] + grid_fee) * model.power_from_grid[i] * time_interval/60)

    model.obj = pyo.Objective(
        expr=sum(model.cash_flow[i] for i in t), sense=pyo.maximize)

    # chose solver and solver-specific options
    # opt = pyo.SolverFactory('glpk')
    # opt.options['mipgap'] = 1e-3    # solver option for GLPK: relative gap, default: 0.0
    # opt.options['tmlim'] = 60*30    # solver option for GLPK: timelimit in seconds
    opt = pyo.SolverFactory('cbc')
    opt.options['seconds'] = 60*30   # solver option for CBC: timelimit in seconds

    _ = opt.solve(model, tee=True)  # solve the problem
    # model.pprint()                          # print results in run terminal

    # RESULTS
    # calculate objective for result output
    objective = sum(model.cash_flow[i].value for i in t)
    return objective, pd.DataFrame({
        #    "Time": [df_actor.iat[i, 0] for i in t],
        "load": load,
        "pv": pv,
        #    "sell_prices": sell_prices,
        "buy_prices": buy_prices,
        "from_grid": [model.power_from_grid[i].value for i in t],
        "to_grid": [model.power_to_grid[i].value for i in t],
        "charge": [model.charging_power[i].value for i in t],
        "discharge": [model.discharging_power[i].value for i in t],
        "soc": [model.stored_energy[i].value / capacity for i in t],
    })


if __name__ == "__main__":
    # INPUT DATA
    # read input data from csv nad write to df
    # just as one example
    df_input_data_actor = pd.read_csv(
        '../projects/example_projects/example_project/scenario/actor_residential_1.csv')
    df_input_data_prices = pd.read_csv(
        '../projects/example_projects/example_project/scenario/MarketMaker.csv')

    objective, df_results = optimize_schedule(df_input_data_actor, df_input_data_prices)
    # PRINTS
    print("RESULTS:")
    print("Objective:", objective)
    pd.set_option('display.max_columns', None)
    print(df_results)

    import matplotlib.pyplot as plt
    f, axs = plt.subplots(3)
    energy_cols = ["load", "pv", "from_grid", "to_grid", "charge", "discharge"]
    df_results[energy_cols].plot(ax=axs[0])
    bat_cols = ["soc"]
    df_results[bat_cols].plot(ax=axs[1])
    price_cols = ["buy_prices"]
    df_results[price_cols].plot(ax=axs[2])
    plt.show()
