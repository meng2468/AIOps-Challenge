import numpy as np
import pandas as pd
import os

from sklearn import preprocessing

pd.options.mode.chained_assignment = None  

def load_esb(data_path):
    df = pd.read_csv(data_path+'esb.csv')
    return df

def normalise(df, name):
    df.loc[:,name] = preprocessing.scale(df[name].values)
    return df

def gen_train_seq(values, time_step):
    output = []
    for i in range(len(values) - time_step):
        output.append(values[i : (i + time_step)])
        
    return np.stack(output)

def get_esb_train_data(df, name, time_step):
    print('-'*60)
    df = df[[name]]
    if len(df.values) - time_step <= 0:
        print('Not enough data!')
        x_train = []
    else:
        df = normalise(df, name)
        x_train = gen_train_seq(df.values, time_step)
    return x_train      