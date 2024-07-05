# REQUIREMENTS# pyomo package needs to be installed
# a solver needs to be installed: CBC or GLPK are open source; for CBC
# coppy cbc_solver in the directory where the repo clone is

import pyomo.environ as pyo
import pandas as pd


def optimize_schedule(df_actor, buy_prices, sell_prices, capacity=10, max_c_rate=1, soc_initial=0.5,
                      ev_capacity=0, ev_max_c_rate=1, ev_soc_initial=0,
                      ts_per_hour=1, end_min_soc=0.6, grid_connection_capacity=20):
    """
    Parametrisation of battery (no prefix) and electric vehicle (prefix: ev_*).
    - df energy values (not power)
    - buy_prices ... including fees
    - capacity=10, max_c_rate=1, soc_initial=0.5 (parametrisation of battery)
    - ts_per_hour minutes per time step
    - end_min_soc: regarding the prediction horizon, the minimal end soc is a fix point
        in order to promote a tendency to not extremely drain batteries at horizon boundary
    - grid_connection_capacity: maximum power drawn from or fed into grid (defaults to 20)
    """
    # TODO battery-efficiency ?

    # single time series vectors
    t = list(range(len(df_actor)))
    # convert from dataFrame to lists and from energy to power
    load = df_actor.loc[:, "load"].mul(ts_per_hour).to_list()
    pv = df_actor.loc[:, "pv"].mul(ts_per_hour).to_list()
    buy_prices = buy_prices.to_list()
    sell_prices = sell_prices.to_list()

    # TODO add electric vehicle
    if ev_capacity != 0:
        ev_avail = df_actor.loc[:, "ev_avail"].to_list()
        ev_demand = df_actor.loc[:, "ev_demand"].mul(ts_per_hour).to_list()
    else:
        ev_avail = [1] * len(df_actor)
        ev_demand = [0] * len(df_actor)

    # PYOMO OPTIMISATION MODEL
    model = pyo.ConcreteModel()
    # Stationary Battery
    model.charging_power = pyo.Var(t, bounds=(0, capacity * max_c_rate))
    model.discharging_power = pyo.Var(t, bounds=(0, capacity * max_c_rate))
    model.stored_energy = pyo.Var(t, bounds=(0, capacity))
    # EV
    model.ev_charging_power = pyo.Var(t, bounds=(0, ev_capacity * ev_max_c_rate))
    model.ev_discharging_power = pyo.Var(t, bounds=(0, ev_capacity * ev_max_c_rate))
    model.ev_stored_energy = pyo.Var(t, bounds=(0, ev_capacity))
    # 0/ 1 for distinguishing between charging / discharging
    model.bi_charge = pyo.Var(t, within=pyo.Binary)
    model.ev_bi_charge = pyo.Var(t, within=pyo.Binary)
    # Exchange with grid
    # - if no upper bound is provided the problem turns out to be unbounded
    model.power_from_grid = pyo.Var(t, bounds=(0, grid_connection_capacity))
    model.power_to_grid = pyo.Var(t, bounds=(0, grid_connection_capacity))
    model.cash_flow = pyo.Var(t)

    #################################
    # energy balance for the system #
    #################################
    model.energy_balance_system = pyo.ConstraintList()
    for i in range(len(t) - 1):
        model.energy_balance_system.add(
            0 ==
            model.power_from_grid[i] + pv[i]
            + model.discharging_power[i] + model.ev_discharging_power[i]
            - load[i] - model.power_to_grid[i] - ev_demand[i]
            - model.charging_power[i] - model.ev_charging_power[i]
        )

    ############################
    # component energy storage #
    ############################
    model.energy_balance_storage = pyo.ConstraintList()
    for i in range(len(t) - 1):
        # constraint
        model.energy_balance_storage.add(
            model.stored_energy[i+1] - model.stored_energy[i] ==
            (model.charging_power[i] - model.discharging_power[i]) / ts_per_hour)

    model.start_storage = pyo.Constraint(
        expr=model.stored_energy[0] ==
        soc_initial * capacity)

    # optional: equal soc at first and last time step
    # (0) no constraint for last time step
    # (1) equal soc at first and last time step:
    #      expr=model.stored_energy[len(t)-1] == model.stored_energy[0]
    # (2) last time step above specific soc
    model.start_end_storage = pyo.Constraint(
        expr=model.stored_energy[len(t)-1] >= capacity * end_min_soc)

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

    ############################
    # component mobile storage #
    ############################
    model.ev_energy_balance_storage = pyo.ConstraintList()
    for i in range(len(t) - 1):
        # constraint
        # ev demand directly incorporated in ev internal balance
        model.ev_energy_balance_storage.add(
            model.ev_stored_energy[i + 1] - model.ev_stored_energy[i] ==
            (model.ev_charging_power[i] - model.ev_discharging_power[i])
            / ts_per_hour)

        model.ev_energy_balance_storage.add(
            model.ev_stored_energy[i] >= ev_demand[i + 1])

    model.ev_start_storage = pyo.Constraint(
        expr=model.ev_stored_energy[0]
        == ev_soc_initial * ev_capacity)

    # optional:
    # (0) no constraint for last time step
    # (1) equal soc at first and last time step:
    #      expr = model.ev_stored_energy[len(t) - 1] == model.ev_stored_energy[0]
    # (2) last time step above specific soc
    model.ev_start_end_storage = pyo.Constraint(
        expr=model.ev_stored_energy[len(t) - 1] >= ev_capacity * end_min_soc)

    # needed in order to not have a discharge that affects the timestep after the last considered
    model.ev_end_no_discharge_storage = pyo.Constraint(
        expr=0 == model.ev_discharging_power[len(t) - 1])

    # binary variable to separate charging and discharging timesteps in order to
    # exclude having both at the same time
    model.ev_binary_charge_storage = pyo.ConstraintList()
    model.ev_binary_discharge_storage = pyo.ConstraintList()
    for i in t:
        model.ev_binary_charge_storage.add(
            model.ev_charging_power[i] <= model.ev_bi_charge[i] * ev_capacity * ev_max_c_rate)
        # Only discharging possible during away-time (not availability)
        if ev_avail[i] == 0:
            # no local charging possible
            model.ev_binary_charge_storage.add(model.ev_charging_power[i] == 0)
            model.ev_binary_discharge_storage.add(
                model.ev_discharging_power[i] == ev_demand[i] * ts_per_hour)

    for i in t:
        model.ev_binary_discharge_storage.add(
            model.ev_discharging_power[i] <=
            (1 - model.ev_bi_charge[i]) * ev_capacity * ev_max_c_rate)

    ##########################################
    # costs to be used in objective function #
    ##########################################
    model.cash_flow_equation = pyo.ConstraintList()
    for i in t:
        model.cash_flow_equation.add(
            model.cash_flow[i] ==
            (
                sell_prices[i] * model.power_to_grid[i]
                - buy_prices[i] * model.power_from_grid[i]
             ) / ts_per_hour)

    model.obj = pyo.Objective(
        expr=sum(model.cash_flow[i] for i in t), sense=pyo.maximize)

    # chose solver and solver-specific options
    # opt = pyo.SolverFactory('glpk')
    # opt.options['mipgap'] = 1e-3    # solver option for GLPK: relative gap, default: 0.0
    # opt.options['tmlim'] = 60*30    # solver option for GLPK: timelimit in seconds
    opt = pyo.SolverFactory('cbc')
    opt.options['seconds'] = 60*30   # solver option for CBC: timelimit in seconds

    # solve the problem
    _ = opt.solve(
        model,
        tee=False  # suppress solver output
    )
    # model.pprint()                          # print results in run terminal

    # RESULTS
    # calculate objective for result output
    try:
        objective = sum(model.cash_flow[i].value for i in t)
    except TypeError:
        from pyomo.util.infeasible import log_infeasible_constraints
        log_infeasible_constraints(model)
        print(model)
    return objective, pd.DataFrame({
        "Time": [df_actor.iat[i, 0] for i in t],
        "load": load,
        "pv": pv,
        "sell_prices": sell_prices,
        "buy_prices": buy_prices,
        "from_grid": [model.power_from_grid[i].value for i in t],
        "to_grid": [model.power_to_grid[i].value for i in t],
        "charge": [model.charging_power[i].value for i in t],
        "discharge": [model.discharging_power[i].value for i in t],
        "soc": [model.stored_energy[i].value / capacity for i in t],
        "ev_charge": [model.ev_charging_power[i].value for i in t],
        "ev_discharge": [model.ev_discharging_power[i].value for i in t],
        "ev_availability": [ev_avail[i] for i in t],
        "ev_demand": [ev_demand[i] for i in t],
        "ev_soc": [0 if ev_capacity == 0 else model.ev_stored_energy[i].value / ev_capacity
                   for i in t],
    })


def plot_optimization_results(df_results):
    import matplotlib.pyplot as plt
    import numpy as np

    # Define color scheme for different energy components
    color_scheme = {
        "load": "red",  # Energy consumption (outflow)
        "ev_demand": "black",  # Energy consumption from EV (outflow)
        "pv": "yellow",  # Energy generation (inflow)
        "from_grid": "blue",  # Energy imported (inflow)
        "to_grid": "purple",  # Energy exported (outflow)
        "charge": "orange",  # Energy going into storage (outflow)
        "discharge": "orange",  # Energy coming out of storage (inflow)
        "ev_charge": "cyan",  # Energy going into storage (outflow)
        "ev_discharge": "cyan",  # Energy coming out of storage (inflow)
    }

    # Assuming df_results is already loaded with your data
    # energy_cols
    # = ["pv", "from_grid", "charge", "ev_charge", "load", "to_grid", "discharge", "ev_discharge"]
    bat_cols = ["soc", "ev_soc"]
    price_cols = ["buy_prices", "sell_prices"]

    # Adjust dataframe for directional flow
    df_adjusted = df_results.copy()
    df_adjusted["load"] = -df_adjusted["load"]  # Negative for visualization
    df_adjusted["ev_demand"] = -df_adjusted["ev_demand"]  # Negative for visualization
    df_adjusted["to_grid"] = -df_adjusted["to_grid"]  # Negative for visualization
    df_adjusted["charge"] = -df_adjusted["charge"]  # Negative for visualization
    df_adjusted["ev_charge"] = -df_adjusted["ev_charge"]  # Negative for visualization

    # Create figure with three subplots
    f, axs = plt.subplots(3, 1, figsize=(12, 18))

    # Inflows and outflows
    inflows = ["pv", "from_grid", "discharge", "ev_discharge"]
    outflows = ["load", "to_grid", "charge", "ev_charge", "ev_demand"]
    cumulative_inflow = np.zeros(len(df_adjusted))
    cumulative_outflow = np.zeros(len(df_adjusted))

    # Plot inflows and outflows using step and fill_between
    for col in inflows:
        axs[0].step(df_adjusted.index, cumulative_inflow + df_adjusted[col], where='post',
                    label=col, color=color_scheme[col])
        axs[0].fill_between(df_adjusted.index, cumulative_inflow,
                            cumulative_inflow + df_adjusted[col], step='post',
                            color=color_scheme[col])
        cumulative_inflow += df_adjusted[col]

    for col in outflows:
        axs[0].step(df_adjusted.index, cumulative_outflow + df_adjusted[col], where='post',
                    label=col, color=color_scheme[col])
        axs[0].fill_between(df_adjusted.index, cumulative_outflow,
                            cumulative_outflow + df_adjusted[col], step='post',
                            color=color_scheme[col])
        cumulative_outflow += df_adjusted[col]

    axs[0].set_title('Household (Inflows & Outflows)')
    axs[0].set_ylabel('Energy (kWh)')
    axs[0].legend(loc='upper right')

    # Plotting the state of charge of the battery
    df_results[bat_cols].mul(100).plot(ax=axs[1], color=['orange', 'cyan'], linewidth=2)
    axs[1].set_title('Battery State of Charge')
    axs[1].set_ylabel('State of Charge (%)')
    axs[1].legend(loc='upper right')

    axs[1].fill_between(
        df_results.index, 0, 100,
        where=(df_results.iloc[df_results.index]["ev_availability"] == 1) | (
                    df_results.iloc[df_results.index]["ev_availability"].diff() == -1),
        interpolate=False, step='pre', color='C2', alpha=0.3)

    # Plotting the buying prices of energy using step
    axs[2].step(df_results.index, df_results[price_cols], where='post', color='green', linewidth=2)
    axs[2].set_title('Energy Purchase Prices')
    axs[2].set_ylabel('Price (EUR/kWh)')
    axs[2].legend(loc='upper right')

    # Adjust layout
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # INPUT DATA
    # read input data from csv nad write to df
    # just as one example
    df_input_data_actor = pd.read_csv(
        '../projects/example_projects/example_project/scenario/actor_residential_1.csv')
    df_input_data_prices = pd.read_csv(
        '../projects/example_projects/example_project/scenario/MarketMaker.csv')
    # include grid fee for buying electricity, see config.cfg
    grid_fee = 0.09
    ev_capacity = max(df_input_data_actor.get("ev_demand", [0]))

    buy_prices = df_input_data_prices.loc[:, "all_buy_prices"] + grid_fee
    sell_prices = df_input_data_prices.loc[:, "all_sell_prices"]

    objective, df_results = optimize_schedule(
        df_input_data_actor, buy_prices, sell_prices,
        capacity=3, max_c_rate=1, soc_initial=0.5,
        ev_capacity=ev_capacity,
        ev_max_c_rate=1,
        ev_soc_initial=0.5)

    # PRINTS
    print("RESULTS:")
    print("Objective:", objective)
    pd.set_option('display.max_columns', None)
    print(df_results)
    plot_optimization_results(df_results)
