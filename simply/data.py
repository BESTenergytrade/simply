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
with open('..\data\v_opendata.json','r') as f:
    load_curve_industry = json.loads(f.read())
# TODO harmonize resolution - to 60 min?
# TODO come up with a mechanism to add noise or randomize load curves