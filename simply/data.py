import pandas
import json
# load household data
# TODO change to datetime format
load_curve = pandas.read_csv('..\data\load_curve_family_with_2_children_1_at_work_1_at_home.csv',
                             sep = ';')
# %%load commercial buildings
# TODO check data types (date + time to datetime), merge columns for total load
load_curve_com  = pandas.read_csv('..\data\load_curve_office-like_businesses.csv',
                                  sep = ';',
                                  low_memory=False) # for uniform dtype
# %%load industry time series
data = json.loads(elevations)
df = pd.json_normalize(data['results'])
