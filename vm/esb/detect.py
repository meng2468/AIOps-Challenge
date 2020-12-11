import pandas as pd
import numpy as np

from statsmodels.tsa.seasonal import STL

def get_onehot(df):
    df['start_time'] = pd.to_datetime(df['startTime'], unit='ms', origin='unix')
    df = df.sort_values(by='start_time')
    df = df.set_index('start_time')
    return df

def get_resid(df):
    stl = STL(df['avg_time'], period=8, robust=True)
    res_avgt = stl.fit()

    stl = STL(df['num'], period=8, robust=True)
    res_num = stl.fit()

    return res_avgt.resid, res_num.resid

def num_anom(resid):
    upper_bound = 158
    lower_bound = -177
    
    if resid[-1] >= upper_bound or resid[-1] <= lower_bound:
        return True
    return False

def avgt_anom(resid):
    upper_bound = .321
    lower_bound = -.144
    
    if resid[-1] >= upper_bound or resid[-1] <= lower_bound:
        return True

    return False

def find_anom(df):
    df = get_onehot(df)
    avgt_resid, num_resid = get_resid(df)

    is_avgt = avgt_anom(avgt_resid)
    is_num = num_anom(num_resid)
    is_sr = df['succee_rate'].values[-1] < 1

    return is_avgt or is_num or is_sr