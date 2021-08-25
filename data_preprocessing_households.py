import glob
import os
import pandas as pd
import random


def preprocess_household_data(dir_in,
                              dir_out,
                              n_files,
                              t_step,
                              n_days,
                              t_resample):

    # merge directory path and relative path
    path_dir = os.path.join(os.getcwd(), dir_in)

    # read all names from directory
    filenames = glob.glob(os.path.join(path_dir, '*.csv'))

    # read some files and resample and save
    filenames = random.sample(filenames, n_files)
    for filename in filenames:
        df = pd.read_csv(filename,
                         sep=';',
                         nrows=n_days*24*60,
                         parse_dates=['Time'],
                         dayfirst=True)
        df = df.set_index('Time')
        df = df.drop(columns='Electricity.Timestep')
        df = df.resample(t_resample).sum()
        filename_out = '_'.join([os.path.basename(filename)[:-4],
                                'sample.csv'])
        path_out_complete = os.path.join(dir_out, filename_out)
        df.to_csv(path_out_complete, index=True, sep=';')


if __name__ == "__main__":
    preprocess_household_data(dir_in=r'simply\data\households',
                              dir_out=r'simply\data\households_sample',
                              n_files=10,
                              t_step=60*15,
                              n_days=7,
                              t_resample="15min")
