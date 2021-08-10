import glob
import os
import pandas as pd
import random

# input path
path_in = 'C:\\Users\daniel.busch\simply\simply\data\households'

# output path 
path_out = 'C:\\Users\daniel.busch\simply\simply\data\households_sample'

# how many files
n = 10

# timestep in seconds
t_step = 60*15 # 15 minutes

# how many days
days = 7

# read all names from directory
filenames = glob.glob(os.path.join(path_in, '*.csv'))

# df_timesteps = pd.read_csv(filenames[0],
#                                usecols = ['Time'],
#                                parse_dates = [0],
#                                sep = ';')

# read some files and resample and save
filenames = random.sample(filenames, n)
for filename in filenames:
        df = pd.read_csv(filename, sep = ';', nrows = days*24*60, parse_dates = ['Time'], dayfirst = True)
        df = df.set_index('Time')
        df = df.drop(columns='Electricity.Timestep')
        df = df.resample('15min').sum()
        filename_out = '_'.join([os.path.basename(filename)[:-4],'sample.csv'])
        path_out_complete = os.path.join(path_out, filename_out)
        df.to_csv(path_out_complete, index=True, sep=';')
        
