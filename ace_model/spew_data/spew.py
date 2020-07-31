import pandas as pd
import numpy as np


pfile_suf = 'people.txt'
hfile_suf = 'households.txt'

race_spew_to_brfss = {1:1, 2:2, 9:5}

income_cat = [15000, 24999, 34999, 49999]
def cat_income(hh_income):
    if pd.isnull(hh_income):
        return 9
    for i in range(len(income_cat), 0, -1):
        if hh_income >= income_cat[i-1]:
            return i + 1
    return 1
    

def cdn_data(data_pre, age = 18):
    pdata_file = data_pre + 'people.txt'
    hdata_file = data_pre + 'households.txt'
    
    pdata = pd.read_csv(pdata_file, dtype = {'sp_id': object, 
                                             'sp_hh_id': object})
    hdata = pd.read_csv(hdata_file, dtype = {'sp_id': object})
    
    
    pdata = pdata[['sp_id', 'sp_hh_id', 'age', 'sex', 
              'race', 'sporder', 'relate']]
    
    hdata = hdata[['sp_id', 'hh_income']]
    
    phdata = pd.merge(pdata, hdata, 
                      left_on=['sp_hh_id'], 
                      right_on = ['sp_id'])
    phdata = phdata[phdata['age'] < 18]
    phdata['_RACE_G1'] = phdata['race'].apply(
        lambda x: race_spew_to_brfss[x] if x in race_spew_to_brfss else 4)
    phdata['_INCOMG'] = phdata['hh_income'].apply(cat_income)
    
    return phdata