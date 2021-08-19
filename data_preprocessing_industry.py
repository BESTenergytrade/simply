import pandas as pd
import json
import os
import numpy as np

# input path
directory = 'C:\\Users\daniel.busch\simply\simply\data\industry'
filename = 'v_opendata.json'
path_in = os.path.join(directory, filename)

# output path
path_out = 'C:\\Users\daniel.busch\simply\simply\data\industry_sample'

# how many days
days = 7

# timesteps is in h
# TODO! this should be automatically detected when resampling (use pd.groupby)
timesteps_h = 0.25

# load industry time series from json file
with open(path_in, 'r') as f:
    load_curve_industry = json.loads(f.read())

# load names and yearly energy consumption from csv file
# TODO! use names as file names when saving the csv
df_ind = pd.read_csv(os.path.join(directory, 'data_description.csv'),
                     sep=';')

# define energy (electricity) consumption for small and medium entreprises
df_consumption = pd.DataFrame({'minE': [1e4, 3e4, 1e5, 2.5e5, 1e6, 5e6],
                               'maxE': [3e4, 1e5, 2.5e5, 1e6, 5e6, 8e6]})

# this dataset uses 60 min timesteps
nb_ts = 24*365
time_idx = pd.date_range("2021-01-01", freq="H", periods=nb_ts)
# save each indsutry sector as csv file
for i, data in enumerate(load_curve_industry[0]['data']):
    df = pd.DataFrame({'Time': time_idx, 'load': data['values']})  # create df
    df = df.set_index('Time')   # set index
    df = df.iloc[0:days*24, 0]  # choose time frame
    rs = df.resample('15min').pad()  # resample, in this case upsample
    df = rs * timesteps_h  # divide by timestep ratio to sustain energy balance
    industry_type = data['internal_id'][0]  # get industry type
    # assign a yearly energy consumption to the normalized load
    # the data is taken from the csv file data_description.csv
    # some industrial companies can have very differnet sizes
    # these get a yearly energy consumption from this study:
    # https://www.eoptimum.de/energie-news/vom-wollen-und-koennen-rund-um-energieeinsparungen-in-unternehmen-12/ 
    if np.isnan(df_ind['yearly consumption [kWh]'].iloc[industry_type-1]):
        energy_category = np.random.choice(np.arange(0, 6),
                                           p=[0.29, 0.3, 0.25,
                                              0.13, 0.02, 0.01])
        yearly_energy = np.random.randint(df_consumption['minE'].iloc[0],
                                          df_consumption['maxE'].iloc[0])
    # some industrial companies usually need a lot of energy
    # they are assigned a value from research +- 10%
    else:
        yearly_energy_fixed = df_ind[
            'yearly consumption [kWh]'].iloc[industry_type-1]
        yearly_energy = np.random.randint(0.9*yearly_energy_fixed,
                                          1.1*yearly_energy_fixed)
    # multiply with yearly energy consumption
    df = df * yearly_energy
    # save file
    filename_out = '_'.join([str(industry_type), 'sample.csv'])
    path_out_complete = os.path.join(path_out, filename_out)
    df.to_csv(path_out_complete, index=True, sep=';')
