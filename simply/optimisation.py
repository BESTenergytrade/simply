# REQUIREMENTS# pyomo package needs to be installed
# a solver needs to be installed: CBC or GLPK are open source; for CBC
# coppy cbc_solver in the directory where the repo clone is
import warnings

import pyomo.environ as pyo
import pandas as pd


def fill_model(model, data):
    # scalar parameters
    model.capacity.set_value(data["capacity"])
    model.soc_initial.set_value(data["soc_initial"])
    model.end_min_soc.set_value(data["end_min_soc"])
    model.max_c_rate.set_value(data["max_c_rate"])

    model.ev_capacity.set_value(data["ev_capacity"])
    model.ev_soc_initial.set_value(data["ev_soc_initial"])
    model.ev_min_soc.set_value(data["ev_min_soc"])
    model.ev_max_power.set_value(data["ev_max_power"])
    model.ev_max_charger.set_value(data["ev_max_charger"])

    model.grid_connection_capacity.set_value(data["grid_connection_capacity"])
    model.ts_per_hour.set_value(data["ts_per_hour"])

    # time series values
    for i in model.T:
        model.load_demand[i] = data["load"][i]
        model.pv[i] = data["pv"][i]
        model.buy_prices[i] = data["buy_prices"][i]
        model.sell_prices[i] = data["sell_prices"][i]
        model.ev_avail[i] = data["ev_avail"][i]
        model.ev_demand[i] = data["ev_demand"][i]


def init_base_model(t_len):
    # PYOMO OPTIMISATION MODEL
    m = pyo.ConcreteModel()

    # sets
    m.T = pyo.RangeSet(0, t_len - 1)

    # --- Parameters (all mutable for clone)
    def zero(model, i): return 0.0

    # scalar parameters
    m.capacity = pyo.Param(initialize=1.0, mutable=True)
    m.soc_initial = pyo.Param(initialize=0.5, mutable=True)
    m.end_min_soc = pyo.Param(initialize=0.2, mutable=True)
    m.max_c_rate = pyo.Param(initialize=1.0, mutable=True)

    m.ev_capacity = pyo.Param(initialize=1.0, mutable=True)
    m.ev_soc_initial = pyo.Param(initialize=0.5, mutable=True)
    m.ev_min_soc = pyo.Param(initialize=0.1, mutable=True)
    m.ev_max_power = pyo.Param(initialize=1.0, mutable=True)
    m.ev_max_charger = pyo.Param(initialize=1.0, mutable=True)

    m.grid_connection_capacity = pyo.Param(initialize=1.0, mutable=True)
    m.ts_per_hour = pyo.Param(initialize=1.0, mutable=True)

    # time series
    m.load_demand = pyo.Param(m.T, initialize=zero, mutable=True)
    m.pv = pyo.Param(m.T, initialize=zero, mutable=True)
    m.ev_demand = pyo.Param(m.T, initialize=zero, mutable=True)
    m.ev_avail = pyo.Param(m.T, initialize=zero, mutable=True)

    m.buy_prices = pyo.Param(m.T, initialize=zero, mutable=True)
    m.sell_prices = pyo.Param(m.T, initialize=zero, mutable=True)

    # --- Variables
    # Stationary Battery
    m.charging_power = pyo.Var(m.T, bounds=(0, m.capacity * m.max_c_rate))
    m.discharging_power = pyo.Var(m.T, bounds=(0, m.capacity * m.max_c_rate))
    m.stored_energy = pyo.Var(m.T, bounds=(0, m.capacity))
    # EV
    m.ev_charging_power = pyo.Var(m.T, bounds=(0, m.ev_max_power))
    m.ev_discharging_power = pyo.Var(m.T, bounds=(0, m.ev_max_power))
    m.ev_stored_energy = pyo.Var(m.T, bounds=(0, m.ev_capacity))
    # 0/ 1 for distinguishing between charging / discharging
    m.bi_charge = pyo.Var(m.T, within=pyo.Binary)
    m.ev_bi_charge = pyo.Var(m.T, within=pyo.Binary)
    # Exchange with grid
    # - if no upper bound is provided the problem turns out to be unbounded
    m.power_from_grid = pyo.Var(m.T, bounds=(0, m.grid_connection_capacity))
    m.power_to_grid = pyo.Var(m.T, bounds=(0, m.grid_connection_capacity))
    m.cash_flow = pyo.Var(m.T)

    # --- Constraints

    #################################
    # energy balance for the system #
    #################################
    m.energy_balance_system = pyo.ConstraintList()
    for i in range(t_len - 1):
        m.energy_balance_system.add(
            0 ==
            m.power_from_grid[i] + m.pv[i]
            + m.discharging_power[i] + m.ev_discharging_power[i]
            - m.load_demand[i] - m.power_to_grid[i] - m.ev_demand[i]
            - m.charging_power[i] - m.ev_charging_power[i]
        )

    ############################
    # component energy storage #
    ############################
    m.energy_balance_storage = pyo.ConstraintList()
    for i in range(t_len - 1):
        # constraint
        m.energy_balance_storage.add(
            m.stored_energy[i + 1] - m.stored_energy[i] ==
            (m.charging_power[i] - m.discharging_power[i]) / m.ts_per_hour)

    m.start_storage = pyo.Constraint(
        expr=m.stored_energy[0] ==
        m.soc_initial * m.capacity)

    # optional: equal soc at first and last time step
    # (0) no constraint for last time step
    # (1) equal soc at first and last time step:
    #      expr=m.stored_energy[t_len-1] == m.stored_energy[0]
    # (2) last time step above specific soc
    m.start_end_storage = pyo.Constraint(
        expr=m.stored_energy[t_len - 1] >= m.capacity * m.end_min_soc)

    # needed in order to not have a discharge that affects the timestep after the last considered
    m.end_no_discharge_storage = pyo.Constraint(
        expr=0 == m.discharging_power[t_len - 1])

    # binary variable to separate charging and discharging timesteps in order to
    # exclude having both at the same time
    m.binary_charge_storage = pyo.ConstraintList()
    for i in m.T:
        m.binary_charge_storage.add(
            m.charging_power[i] <= m.bi_charge[i] * m.capacity * m.max_c_rate)

    m.binary_discharge_storage = pyo.ConstraintList()
    for i in m.T:
        m.binary_discharge_storage.add(
            m.discharging_power[i] <= (1 - m.bi_charge[i]) * m.capacity * m.max_c_rate)

    ############################
    # component mobile storage #
    ############################
    m.ev_energy_balance_storage = pyo.ConstraintList()
    for i in range(t_len - 1):
        # constraint
        # ev demand directly incorporated in ev internal balance
        m.ev_energy_balance_storage.add(
            m.ev_stored_energy[i + 1] - m.ev_stored_energy[i] ==
            (m.ev_charging_power[i] - m.ev_discharging_power[i])
            / m.ts_per_hour)

        # The energy to be consumed during the succeeding time slot (while not available) has to be
        # stored at the end of the current time slot, i.e. at the beginning of the next time slot
        # try to keep a minimal soc after trip
        # - ev_demand was converted to a power above
        # if ev_demand[i + 1] != 0: #without condition not the whole capacity is usable
        m.ev_energy_balance_storage.add(
            m.ev_stored_energy[i + 1] >=
            m.ev_demand[i + 1] / m.ts_per_hour + m.ev_min_soc * m.ev_capacity
        )

    m.ev_start_storage = pyo.Constraint(
        expr=m.ev_stored_energy[0]
        == m.ev_soc_initial * m.ev_capacity)

    # optional:
    # (0) no constraint for last time step
    # (1) equal soc at first and last time step:
    #      expr = m.ev_stored_energy[t_len - 1] == m.ev_stored_energy[0]
    # (2) last time step above specific soc
    #     Infeasible, if last timeslot has availability = 0
    #     TODO Multiple timeslots might be necessary to get to a specific soc
    #      or ev_capacity >= max(ev_demand)/ / ts_per_hour / (1 - end_min_soc)
    # m.ev_start_end_storage = pyo.Constraint(
    #    expr=m.ev_stored_energy[t_len - 1] >= ev_capacity-max(ev_demand) / ts_per_hour
    # )

    # needed in order to not have a discharge that affects the timestep after the last considered
    m.ev_end_no_discharge_storage = pyo.Constraint(
        expr=0 == m.ev_discharging_power[t_len - 1] - m.ev_demand[t_len - 1])

    # binary variable to separate charging and discharging timesteps in order to
    # exclude having both at the same time
    m.ev_binary_charge_storage = pyo.ConstraintList()
    m.ev_binary_discharge_storage = pyo.ConstraintList()
    """ (toggle previous/new constraints by inserting # before quotation in this line
    # START previous constaints: EV charge/discharge
    for i in m.T:
        if ev_avail[i] == 1:
            m.ev_binary_charge_storage.add(
                m.ev_charging_power[i] <= m.ev_bi_charge[i] * m.ev_max_charger)
        elif ev_avail[i] == 0:
            # no local charging possible
            # Only discharging possible during away-time (i.e. not available)
            m.ev_binary_charge_storage.add(m.ev_charging_power[i] == 0)
            # save consumption energy and discharge with max power
            # until the consumption energy left is below max power
            assert ev_demand[i] <= ev_max_power
            m.ev_binary_discharge_storage.add(
                m.ev_discharging_power[i] == m.ev_demand[i])

    for i in m.T:
        # different discharging limits depending on availability i.e. charger or battery constraint
        if ev_avail[i] == 1:
            m.ev_binary_discharge_storage.add(
                m.ev_discharging_power[i] <=
                (1 - m.ev_bi_charge[i]) * m.ev_max_charger)
        elif ev_avail[i] == 0:
            m.ev_binary_discharge_storage.add(
                m.ev_discharging_power[i] <=
                (1 - m.ev_bi_charge[i]) * m.ev_max_power)
    # END previous constaints: EV charge/discharge
    """
    # START: new constaints: EV charge/discharge
    for i in m.T:
        # charging possible only when available
        m.ev_binary_charge_storage.add(
            m.ev_charging_power[i] <=
            m.ev_bi_charge[i] * m.ev_max_charger * m.ev_avail[i]
        )

        # discharging possible depends on availability
        m.ev_binary_discharge_storage.add(
            m.ev_discharging_power[i] <=
            (1 - m.ev_bi_charge[i]) *
            (m.ev_max_charger * m.ev_avail[i] +  # when available below charger power
             m.ev_demand[i] * (1 - m.ev_avail[i]))  # when not available below/equal demand
        )

        # when not available (above/)equal demand (see above)
        m.ev_binary_discharge_storage.add(
            m.ev_discharging_power[i] >= m.ev_demand[i] * (1 - m.ev_avail[i]))
    # END new constaints: EV charge/discharge"""

    ##########################################
    # costs to be used in objective function #
    ##########################################
    m.cash_flow_equation = pyo.ConstraintList()
    for i in m.T:
        m.cash_flow_equation.add(
            m.cash_flow[i] ==
            (
                m.sell_prices[i] * m.power_to_grid[i]
                - m.buy_prices[i] * m.power_from_grid[i]
            ) / m.ts_per_hour)

    m.obj = pyo.Objective(
        expr=sum(m.cash_flow[i] for i in m.T), sense=pyo.maximize)

    return m


def optimize_schedule(
        df_actor, buy_prices, sell_prices, capacity=10, max_c_rate=1, soc_initial=0.5,
        ev_capacity=0, ev_max_c_rate=1, ev_soc_initial=0, ev_min_soc=0.1,
        charger_max_power=11, end_min_soc=0.6, grid_connection_capacity=20,
        ts_per_hour=1, model=None, actor_id=None):
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
    :param ev_min_soc: minimal state of charge  (electric vehicle parameter) default=0.1
    :param charger_max_power: maximum power of charger for ev (electric vehicle parameter)
        default=11
    :param end_min_soc: regarding the prediction horizon, the minimal end soc is a fix point
        in order to promote a tendency to not extremely drain batteries at the end of the horizon
    :param grid_connection_capacity: maximum power drawn from or fed into grid; default=20)
    :param ts_per_hour: time steps per hour;  default=1
    :param model: pyomo optimization model default=None
    """
    # TODO battery-efficiency ?

    # single time series vectors
    t_len = len(df_actor)
    t = list(range(t_len))

    # translate to data dictionary
    data = {}

    # time series values
    data["capacity"] = capacity
    data["max_c_rate"] = max_c_rate
    data["ev_capacity"] = ev_capacity
    data["ev_min_soc"] = ev_min_soc
    data["grid_connection_capacity"] = grid_connection_capacity
    data["ts_per_hour"] = ts_per_hour
    data["soc_initial"] = soc_initial
    data["ev_soc_initial"] = ev_soc_initial
    data["end_min_soc"] = end_min_soc
    # convert from dataFrame to lists and from energy to power
    data["load"] = df_actor.loc[:, "load"].mul(ts_per_hour).to_list()
    data["pv"] = df_actor.loc[:, "pv"].mul(ts_per_hour).to_list()
    if isinstance(buy_prices, pd.Series):
        data["buy_prices"] = buy_prices.to_list()
        data["sell_prices"] = sell_prices.to_list()
    else:
        # already list
        data["buy_prices"] = buy_prices
        data["sell_prices"] = sell_prices

    # Add electric vehicle
    # ev charging/discharing is limited by either the charger or the battery c-rate
    ev_max_power = ev_capacity * ev_max_c_rate
    ev_max_charger = min(charger_max_power, ev_max_power)
    # store these scalars too
    data["ev_max_power"] = ev_max_power
    data["ev_max_charger"] = ev_max_charger

    if ev_capacity != 0:
        data["ev_avail"] = df_actor.loc[:, "ev_avail"].to_list()
        data["ev_demand"] = df_actor.loc[:, "ev_demand"].mul(ts_per_hour).to_list()

        consumption = 0
        for i in range(len(data["ev_demand"])):
            if data["ev_avail"][i] == 0:
                if data["ev_demand"][i] != 0:
                    # save consumption energy and discharge with max power
                    # until the consumption energy left is below max power
                    consumption = consumption + data["ev_demand"][i] / ts_per_hour
                if consumption > 0:
                    # consumption cannot exceed maximum battery power
                    dischargable = min(consumption * ts_per_hour, ev_max_power)
                    # update demand and carry along the rest
                    data["ev_demand"][i] = dischargable
                    consumption -= dischargable / ts_per_hour
            else:
                # If consumption variable is not 0 => the availability ended to be 0 (driving)
                # before it was possible to discharge with max power according to c-rate
                assert consumption == 0, (
                    "EV consumed more energy than c-rate allows it during non-availability"
                )
    else:
        data["ev_avail"] = [1] * len(df_actor)
        data["ev_demand"] = [0] * len(df_actor)

    # PYOMO OPTIMISATION MODEL
    if model is None:
        model = init_base_model(t_len)

    fill_model(model, data)
    optimize_run(model)

    # RESULTS
    # calculate objective for result output
    try:
        objective = sum(model.cash_flow[i].value for i in model.T)
    except TypeError:
        warnings.warn(f"Actor {actor_id} without optimization result! Will be skipped. Dump/print: values")
        print({
            "actor_id": actor_id,
            "df_actor": df_actor.to_dict(),
            "buy_prices": list(buy_prices),
            "sell_prices": list(sell_prices),
            "capacity": capacity,
            "max_c_rate": max_c_rate,
            "soc_initial": soc_initial,
            "ev_capacity": ev_capacity,
            "ev_max_c_rate": ev_max_c_rate,
            "ev_soc_initial": ev_soc_initial,
            "charger_max_power": charger_max_power,
            "ts_per_hour": ts_per_hour,
            "end_min_soc": end_min_soc,
            "grid_connection_capacity": grid_connection_capacity
        })
        if ev_capacity != 0:
            print(f"init soc: {ev_soc_initial}, demand_max: "
                  f"{max(data['ev_demand']) / ts_per_hour / ev_capacity}")
        else:
            print("No EV")
        raise TypeError

    return model, objective, pd.DataFrame({
        "Time": [df_actor.iat[i, 0] for i in t],
        "load": [model.load_demand[i].value for i in t],
        "pv": [model.pv[i].value for i in t],
        "sell_prices": [model.sell_prices[i].value for i in t],
        "buy_prices": [model.buy_prices[i].value for i in t],
        "from_grid": [model.power_from_grid[i].value for i in t],
        "to_grid": [model.power_to_grid[i].value for i in t],
        "charge": [model.charging_power[i].value for i in t],
        "discharge": [model.discharging_power[i].value for i in t],
        "soc": [model.stored_energy[i].value / capacity for i in t],
        "ev_charge": [model.ev_charging_power[i].value for i in t],
        "ev_discharge": [model.ev_discharging_power[i].value for i in t],
        "ev_availability": [model.ev_avail[i].value for i in t],
        "ev_demand": [model.ev_demand[i].value for i in t],
        "ev_soc": [0 if ev_capacity == 0 else model.ev_stored_energy[i].value / ev_capacity
                   for i in t],
    })


def optimize_run(model, solver_name='cbc', options={'threads': 4}):
    # chose solver and solver-specific options
    # opt = pyo.SolverFactory('glpk')
    # opt.options['mipgap'] = 1e-3    # solver option for GLPK: relative gap, default: 0.0
    # opt.options['tmlim'] = 60*30    # solver option for GLPK: timelimit in seconds

    opt = pyo.SolverFactory(solver_name)

    opt.options['seconds'] = 1   # solver option for CBC: timelimit in seconds
    # opt.options['tol'] = 0.0001      # solver option for CBC: tolerance
    # opt.options['ratio'] = 0.01

    opt.options['threads'] = options['threads']

    # only for appsi_highs
    opt.options["log_to_console"] = False  # silence solver console log
    opt.options["log_file"] = "solver.log"  # HiGHS writes here

    # solve the problem
    _ = opt.solve(
        model,
        tee=False,  # suppress solver output
    )
    # model.pprint()                          # print results in run terminal


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
    ev_capacity = max(df_input_data_actor.get("ev_demand", [0])) * 1.2

    # market maker buy prices => actor sell prices and vice versa
    sell_prices = df_input_data_prices.loc[:, "all_buy_prices"]
    buy_prices = df_input_data_prices.loc[:, "all_sell_prices"] + grid_fee

    """
    # For debugging:
    parameter_dict =

    parameter_dict["df_actor"] = pd.DataFrame(parameter_dict["df_actor"])
    print(parameter_dict["df_actor"])
    objective, df_results = optimize_schedule(
        **parameter_dict)
    #"""

    _, objective, df_results = optimize_schedule(
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
