import numpy as np
import pandas as pd
import os

from sklearn import preprocessing

def load_dfs(data_path):
    dfs = {}
    for file in os.listdir(data_path):
        print('Saving ' + file[:-4] + ' into dfs')
        dfs[file[:-4]] = pd.read_csv(data_path+file) 
    return dfs

def normalise(df):
    df['value'] = preprocessing.scale(df['value'].values)
    return df

def gen_train_seq(values, time_step):
    output = []
    for i in range(len(values) - time_step):
        output.append(values[i : (i + time_step)])
        
    return np.stack(output)

def get_kpi_train_data(df, time_step):
    print('-'*60)
    print("Getting KPI training data")
    failures = []
    x_train_list = []
    for host in list(df['cmdb_id'].unique()):
        df_host = df[df.cmdb_id==host][['value']]
        if len(df_host.values) - time_step <= 0:
            print(host, ' does not have enough data, skipping!')
            failures.append(host)
        else:
            df_host = normalise(df_host)
            x_train_list.append(gen_train_seq(df_host.values, time_step))
    x_train = np.concatenate(x_train_list)
    print("Not enough data for timestep ", time_step, " in ", failures)
    print(x_train.shape)
    return x_train