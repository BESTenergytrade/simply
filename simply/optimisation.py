# REQUIREMENTS# pyomo package needs to be installed
# a solver needs to be installed: CBC or GLPK are open source; for CBC
# coppy cbc_solver in the directory where the repo clone is

import pyomo.environ as pyo
import pandas as pd


def optimize_schedule(df_actor, buy_prices, sell_prices, capacity=10, max_c_rate=1, soc_initial=0.5,
                      ev_capacity=0, ev_max_c_rate=1, ev_soc_initial=0, charger_max_power=11,
                      ts_per_hour=1, end_min_soc=0.6, grid_connection_capacity=20):
    """
    Optimizes load, pv time series with battery and electric vehicle flexibility based on buying and
    selling price time series.

    Parametrisation of battery (no prefix) and electric vehicle (prefix: ev_*).

    :param df_actor: pd.DataFrame with energy values (not power) with columns 'load', 'pv'
        and optionally 'ev_avail' and 'ev_demand'
    :param buy_prices: price series at which energy can be bought including fees
    :param sell_prices: price series at which energy can be sold
    :param capacity: maximum energy that can be stored (battery parameter) default=10
    :param max_c_rate: How many times the capacity can be (dis-)charged within an hour
        (battery parameter) default=1
    :param soc_initial: initial state of charge (battery parameter) default=0.5
    :param ev_capacity: maximum energy that can be stored (electric vehicle parameter) default=10
    :param ev_max_c_rate: How many times the capacity can be (dis-)charged within an hour
        (electric vehicle parameter) default=1
    :param ev_soc_initial: initial state of charge  (electric vehicle parameter) default=0.5
    :param ts_per_hour: time steps per hour;  default=1
    :param end_min_soc: regarding the prediction horizon, the minimal end soc is a fix point
        in order to promote a tendency to not extremely drain batteries at the end of the horizon
    :param grid_connection_capacity: maximum power drawn from or fed into grid; default=20)
    """
    # TODO battery-efficiency ?

    # single time series vectors
    t = list(range(len(df_actor)))
    # convert from dataFrame to lists and from energy to power
    load = df_actor.loc[:, "load"].mul(ts_per_hour).to_list()
    pv = df_actor.loc[:, "pv"].mul(ts_per_hour).to_list()
    if isinstance(buy_prices, pd.Series):
        buy_prices = buy_prices.to_list()
        sell_prices = sell_prices.to_list()

    # Add electric vehicle
    # ev charging/discharing is limited by either the charger or the battery c-rate
    ev_max_power = ev_capacity * ev_max_c_rate
    ev_max_charger = min(charger_max_power, ev_max_power)
    if ev_capacity != 0:
        ev_avail = df_actor.loc[:, "ev_avail"].to_list()
        ev_demand = df_actor.loc[:, "ev_demand"].mul(ts_per_hour).to_list()
        consumption = 0
        for i in range(len(ev_demand)):
            if ev_avail[i] == 0:
                if ev_demand[i] != 0:
                    # save consumption energy and discharge with max power
                    # until the consumption energy left is below max power
                    consumption = consumption + ev_demand[i] / ts_per_hour
                if consumption > 0:
                    # consumption cannot exceed maximum battery power
                    dischargable = min(consumption * ts_per_hour, ev_max_power)
                    # update demand and carry along the rest
                    ev_demand[i] = dischargable
                    consumption -= dischargable / ts_per_hour
            else:
                # If consumption variable is not 0 => the availability ended to be 0 (driving)
                # before it was possible to discharge with max power according to c-rate
                assert consumption == 0, "EV consumed more energy than c-rate allows it during" \
                                         " non-availability"
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
    model.ev_charging_power = pyo.Var(t, bounds=(0, ev_max_power))
    model.ev_discharging_power = pyo.Var(t, bounds=(0, ev_max_power))
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

        # The energy to be consumed during the succeeding time slot (while not available) has to be
        # stored at the end of the current time slot, i.e. at the beginning of the next time slot
        # try to keep a minimal soc after trip
        # - ev_demand was converted to a power above
        if ev_demand[i + 1] != 0:
            model.ev_energy_balance_storage.add(
                model.ev_stored_energy[i + 1] >=
                # max(ev_capacity * ev_target_soc, ev_demand[i + 1] + 0.1 * ev_capacity)
                ev_demand[i + 1] / ts_per_hour + 0.1 * ev_capacity
            )

    model.ev_start_storage = pyo.Constraint(
        expr=model.ev_stored_energy[0]
        == ev_soc_initial * ev_capacity)

    # optional:
    # (0) no constraint for last time step
    # (1) equal soc at first and last time step:
    #      expr = model.ev_stored_energy[len(t) - 1] == model.ev_stored_energy[0]
    # (2) last time step above specific soc
    #     Infeasible, if last timeslot has availability = 0
    #     TODO Multiple timeslots might be necessary to get to a specific soc
    #      or ev_capacity >= max(ev_demand)/ / ts_per_hour / (1 - end_min_soc)
    # model.ev_start_end_storage = pyo.Constraint(
    #    expr=model.ev_stored_energy[len(t) - 1] >= ev_capacity-max(ev_demand) / ts_per_hour
    # )

    # needed in order to not have a discharge that affects the timestep after the last considered
    model.ev_end_no_discharge_storage = pyo.Constraint(
        expr=0 == model.ev_discharging_power[len(t) - 1] - ev_demand[len(t) - 1])

    # binary variable to separate charging and discharging timesteps in order to
    # exclude having both at the same time
    model.ev_binary_charge_storage = pyo.ConstraintList()
    model.ev_binary_discharge_storage = pyo.ConstraintList()
    for i in t:
        if ev_avail[i] == 1:
            model.ev_binary_charge_storage.add(
                model.ev_charging_power[i] <= model.ev_bi_charge[i] * ev_max_charger)
        elif ev_avail[i] == 0:
            # no local charging possible
            # Only discharging possible during away-time (i.e. not available)
            model.ev_binary_charge_storage.add(model.ev_charging_power[i] == 0)
            # save consumption energy and discharge with max power
            # until the consumption energy left is below max power
            assert ev_demand[i] <= ev_max_power
            model.ev_binary_discharge_storage.add(
                model.ev_discharging_power[i] == ev_demand[i])

    for i in t:
        # different discharging limits depending on availability i.e. charger or battery constraint
        if ev_avail[i] == 1:
            model.ev_binary_discharge_storage.add(
                model.ev_discharging_power[i] <=
                (1 - model.ev_bi_charge[i]) * ev_max_charger)
        elif ev_avail[i] == 0:
            model.ev_binary_discharge_storage.add(
                model.ev_discharging_power[i] <=
                (1 - model.ev_bi_charge[i]) * ev_max_power)

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
    opt.options['seconds'] = 1   # solver option for CBC: timelimit in seconds
    # opt.options['tol'] = 0.0001      # solver option for CBC: tolerance
    opt.options['threads'] = 4
    # opt.options['ratio'] = 0.01
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
        print({
            "df_actor": df_actor.to_dict(),
            "buy_prices": buy_prices,
            "sell_prices": sell_prices,
            "capacity": capacity,
            "max_c_rate": max_c_rate,
            "soc_initial": soc_initial,
            "ev_capacity": ev_capacity,
            "ev_max_c_rate": ev_max_c_rate,
            "ev_soc_initial": ev_soc_initial,
            "ev_max_charger": ev_max_charger,
            "ts_per_hour": ts_per_hour,
            "end_min_soc": end_min_soc,
            "grid_connection_capacity": grid_connection_capacity
        })
        print(f"init soc: {ev_soc_initial}, demand_max: {max(ev_demand)/ts_per_hour/ev_capacity}")

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

    """
    # For debugging:
    parameter_dict =

    parameter_dict["df_actor"] = pd.DataFrame(parameter_dict["df_actor"])
    print(parameter_dict["df_actor"])
    objective, df_results = optimize_schedule(
        **parameter_dict)
    #"""

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
