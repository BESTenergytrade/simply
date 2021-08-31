import data_preprocessing_industry
import pandas as pd
import json

def test_sample_industry_data(path_original_data, # where is the original json file
                              dir_sample_data,  # where is the sampled data
                              start_date,  # start date for the test
                              n_days):  # number of days for test period
    # open original file
    with open(path_original_data, 'r') as f:
        load_curve_industry = json.loads(f.read())
        
    # resample original data
    df_list = data_preprocessing_industry.preprocess_industry_data(
            dir_in=r'simply\data\industry',
            filename = 'v_opendata.json',
            t_step=0.25,
            n_days=14,
            start_date="2021-01-01",
            t_resample="15min")
    
    # create time index for original data (always the same)
    nb_ts = 24*365
    time_idx = pd.date_range("2021-01-01", freq="H", periods=nb_ts)
    # go through all industry types
    for i, data in enumerate(load_curve_industry[0]['data']):
        # extract industry type
        industry_type = data['internal_id'][0]
        # create data frame from original data
        df = pd.DataFrame(
            {'Time': time_idx,
             str(industry_type): data['values'].copy()
             })
        df = df.set_index('Time')   # set index
        # TODO! Ab hier funktioniert es noch nicht wie es soll!
        # compare original and resampled data
        e_original = float(df['2021-01-01 00:00:00':'2021-01-02 00:00:00'].sum())
        e_resample = df_list[i]['2021-01-01 00:00:00':'2021-01-02 00:00:00'].sum()
        
        assert e_original == e_resample, 'energy balance incorrect'
    
    print('everything is ok')
    
if __name__ == "__main__":
    test_sample_industry_data(
        path_original_data= r'simply\data\industry\v_opendata.json',
        dir_sample_data=r'simply\data\industry_sample',
        start_date="2021-01-01",  # must be later than sample start
        n_days=6  # must not exceed end of sample
        )
    
    
