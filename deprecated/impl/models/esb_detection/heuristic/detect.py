import pandas as pd
import numpy as np

from statsmodels.tsa.seasonal import STL

def get_onehot(df):
    df['start_time'] = pd.to_datetime(df['startTime'], unit='ms', origin='unix')
    df = df.sort_values(by='start_time')
    df = df.set_index('start_time')
    return df


def find_anom(df):
    print('*'*40)
    print('Running ESB detection on: ')
    print(df.tail())

    df = get_onehot(df)

    is_sr = df['succee_rate'].values[-1] < 1
    if is_sr:
        print("Success rate < 1")

    ub_time = .92 # max
    lb_time = .49 # min

    is_avgt = False
    curr_t = df['avg_time'].values[-1]
    if curr_t > ub_time or curr_t < lb_time:
        print('Average time crosses threshold')
        is_avgt = True

    ub_num = 650 # ~99 percentile (only happens once in whole set)
    lb_num = 311 # .1 percentile
    is_num = False #
    curr_n = df['num'].values[-1]
    if curr_n > ub_num or curr_n < lb_num:
        print('Num crosses threshold')
        is_num = True

    return is_avgt or is_num or is_sr
