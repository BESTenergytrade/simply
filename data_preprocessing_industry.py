import pandas as pd
import json
import os
import numpy as np

def preprocess_industry_data(dir_in,  # directory of original data
                             filename,  # filename
                             t_step,  # time step in hours (must match t_resample)
                             n_days,  # how many days
                             t_resample):  # '15min', '2min' etc.
# TODO! until now this function can only upsample not downsample the data
# TODO! t_step and t_resample are separate variables even though they contain
# the same information

    # merge directory path and relative path
    path_dir = os.path.join(os.getcwd(), dir_in)  # is this necessary?
    path_in = os.path.join(path_dir, filename)

    # load industry time series from json file
    with open(path_in, 'r') as f:
        load_curve_industry = json.loads(f.read())

    # this dataset uses 60 min timesteps
    nb_ts = 24*365
    time_idx = pd.date_range("2021-01-01", freq="H", periods=nb_ts)
    # save each indsutry sector as df
    df_list = []
    for i, data in enumerate(load_curve_industry[0]['data']):
        industry_type = data['internal_id'][0]  # get industry type
        df = pd.DataFrame(
            {'Time': time_idx,
             str(industry_type): data['values']
             }
            )  # create df
        df = df.set_index('Time')   # set index
        df = df.iloc[0:n_days*24, 0]  # choose time frame
        rs = df.resample(t_resample).pad()  # resample, in this case upsample
        df = rs * t_step  # divide by timestep ratio to sustain energy balance

        df_list.append(df)
    # return as list of dataframes
    return df_list

def scale_industry_data(df_list, directory, categories, probabilities):
    df_list_scale = []
    for df in df_list:
        # choose a category based on the set probabilties
        energy_category = np.random.choice(np.arange(
            0, len(probabilities)), p=probabilities)
        # randomly choose a value in the given range
        yearly_energy = np.random.randint(categories['minE'].iloc[energy_category],
                                          categories['maxE'].iloc[energy_category])
        # multiply with yearly energy consumption
        df = df * yearly_energy
        # append to list
        df_list_scale.append(df)
    # return as list of dataframes
    return df_list_scale

def save_industry_data(df_list, dir_out):
    for df in df_list:
        # generate filename based on industry type
        filename_out = '_'.join([str(df.name), 'sample.csv'])
        # merge directory and filename to path
        path_out_complete = os.path.join(dir_out, filename_out)
        #  save as csv
        df.to_csv(path_out_complete, index=True, sep=';')

if __name__ == "__main__":
    # resample industry data
    industry_data_sample = preprocess_industry_data(
        dir_in=r'simply\data\industry',
        filename = 'v_opendata.json',
        t_step=0.25,
        n_days=7,
        t_resample="15min")
    # scale industry data
    industry_data_sample = scale_industry_data(
        industry_data_sample,
        directory= r'simply\data\industry',
        categories=pd.DataFrame(
                  {'minE': [1e4, 3e4, 1e5, 2.5e5, 1e6, 5e6],
                   'maxE': [3e4, 1e5, 2.5e5, 1e6, 5e6, 8e6]
                  }
                               ),
        probabilities=[0.29, 0.3, 0.25, 0.13, 0.02, 0.01])
    # save industry data
    save_industry_data(df_list=industry_data_sample,
        dir_out=r'simply\data\industry_sample')
