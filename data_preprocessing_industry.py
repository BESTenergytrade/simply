import pandas as pd
import json
import os
# TODO! what is the unit of the data set?
# input path
directory = 'C:\\Users\daniel.busch\simply\simply\data\industry'
filename = 'v_opendata.json'
path_in = os.path.join(directory, filename)

# output path 
path_out = 'C:\\Users\daniel.busch\simply\simply\data\industry_sample'

# how many days
days = 7

# timesteps in h
# TODO! this should be automatically detected when resampling (use pd.groupby)
timesteps_h = 0.25

# load industry time series from json file
with open(path_in,'r') as f:
    load_curve_industry = json.loads(f.read())

# load names from csv file
# TODO! use names as file names when saving the csv
df_names = pd.read_csv(os.path.join(directory, 'data_description.csv'), sep = ';')

# this dataset uses 60 min timesteps
nb_ts = 24*365
time_idx = pd.date_range("2021-01-01", freq="H", periods=nb_ts)
# save each indsutry sector as csv file
for i, data in enumerate(load_curve_industry[0]['data']):
    df = pd.DataFrame({'Time': time_idx, 'load': data['values']}) # create df
    df = df.set_index('Time') # set index
    df = df.iloc[0:days*24, 0] # choose time frame
    rs = df.resample('15min').pad() # resample, in this cas upsample
    df = rs * timesteps_h # divide by timestep ratio to sustain energy balance
    industry_type = str(data['internal_id'][0]) # select industry type
    filename_out = '_'.join([industry_type,'sample.csv'])
    path_out_complete = os.path.join(path_out, filename_out)
    df.to_csv(path_out_complete, index=True, sep=';')