# todo default_grid_fee zusätzlich zu Preisen von Market Maker (nur bei buy)

# REQUIREMENTS
# pyomo needs to be installed: pip install pyomo
# a solver needs to be installed: for CBC coppy CBC.exe in the directory

#from pyomo.environ import *
import pandas as pd

# INPUT DATA
#read input data from csv nad write to df
df_input_data_actor = pd.read_csv('../projects/example_projects/example_project/scenario/actor_residential_1.csv') # just as one example
df_input_data_prices = pd.read_csv('../projects/example_projects/example_project/scenario/MarketMaker.csv')

# generate single time series vectors
load = [0] * len(df_input_data_actor)
pv = [0] * len(df_input_data_actor)
buy_prices = [0] * len(df_input_data_prices)
sell_prices = [0] * len(df_input_data_prices)

# fill time series vectors with data from df
for i in range (len(df_input_data_actor)):
    load[i] = df_input_data_actor.iat[i, 1]
    pv[i] = df_input_data_actor.iat[i, 2]
    buy_prices[i] = df_input_data_prices.iat[i, 1]
    sell_prices[i] = df_input_data_prices.iat[i, 2]

print(df_input_data_actor)
print(load)
print(pv)

print(df_input_data_prices)
print(buy_prices)
print(sell_prices)

"""
# parametrisation of battery specified in battery.py
capacity= 10 # in kWh
max_c_rate=1
soc_initial=0.5


# other parameters
time_interval=15 # minutes per time step


# PYOMO OPTIMISATION MODEL


model = ConcreteModel()
model.charging_power = Var(t, bounds=(0, capacity * max_c_rate))
model.discharging_power = Var(t, bounds=(0, capacity * max_c_rate))
model.stored_energy = Var(t, bounds=(0, capacity))
model.bi_charge = Var(t, within = Binary)
model.cash_flow = Var(t)
model.profit = Var()

model.energy_balance_node = ConstraintList()
for i in range(len(t) - 1):
# todo: Netzbezug / Netzeinspeisung mit Speicherbe- und Entladung verknüpfen
model.energy_balance_storage = ConstraintList()
for i in range(len(t) - 1):
    model.energy_balance_storage.add(model.stored_energy[i+1] - model.stored_energy[i] == (model.charging_power[i] - model.discharging_power[i]) * time_interval/60)                                                          # constraint
model.start_storage = Constraint(expr = model.stored_energy[0] == soc_initial * capacity)
model.start_end_storage = Constraint(expr = model.stored_energy[0] == model.stored_energy[len(t)-1]) # optional: equal soc at first and last time step
model.end_no_discharge = Constraint(expr= 0 == model.discharging_power[len(t) - 1]) # needed in order to not have a discharge that affects the timestep after the last considered
model.binary_charge = ConstraintList() # binary variable to separate charging and discharging timesteps in order to exclude having both at the same time
for i in t:
    model.binary_charge.add(model.charging_power[i] <= model.bi_charge[i] * capacity * max_c_rate)
model.binary_discharge = ConstraintList()
for i in t:
    model.binary_discharge.add(model.discharging_power[i] <= (1 - model.bi_charge[i]) * capacity * max_c_rate)
model.cash_flow_equation = ConstraintList() # todo: ab hier weitermachen
for i in t:
    model.cash_flow_equation.add(model.cash_flow[i] == electricity_price[i] * model.discharging_power[i] - (electricity_price[i] + fee) * model.charging_power[i])

                model.obj = Objective(expr = sum(model.cash_flow[i] for i in t), sense=maximize)

                opt = SolverFactory('glpk')
    
"""