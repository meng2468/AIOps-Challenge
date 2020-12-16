import pandas as pd
import numpy as np
import os

from alibi_detect.od import SpectralResidual

# Helper functions

def get_data():
    data_path = '../../../../data/train_data/host'

    dfs = {}
    for file in os.listdir(data_path):
        print('Reading ' + file)
        dfs[file[:-4]] = pd.read_csv(data_path+'/'+file) 
    return dfs

## Checks for minute-wise data, if a datapoint doesn't exist, the left-next non-null data point is taken
def get_past(df, curr_time, window_size):
    step_size = 1000*60
    df_new = pd.DataFrame(columns=['timestamp', 'value'])

    for time in [curr_time - step_size*x for x in range(window_size)][::-1]:
        if len(df[df.timestamp == time]) == 0:
            plug_time = np.max(df[df.timestamp < time]['timestamp'])
            row = {}
            row['timestamp'] = time
            if len(df.loc[df.timestamp==plug_time, 'value']) == 0:
                row['value'] = 0
            else: 
                row['value'] = df.loc[df.timestamp==plug_time, 'value'].values[0]
            df_new = df_new.append(row, ignore_index=True)
        else:
            row = {}
            row['timestamp'] = time
            row['value'] = df.loc[df.timestamp==time, 'value'].values[0]
            df_new = df_new.append(row, ignore_index=True)
    
    return df_new


dfs = get_data()

window_size = 30

od = SpectralResidual(
    window_amp=window_size,
    window_local=window_size,
    n_est_points=5,
    n_grad_points=5
)

df_thresh = pd.DataFrame(columns=['key','name','host','thresh'])
df_thresh = pd.read_csv('thresh.csv')

for key in dfs:
    print('*'*40)
    print(key)
    df = dfs[key]
    curr_time = np.max([np.max(dfs[k].timestamp.unique()) for k in dfs])
    for name in list(df.name.unique()):
        df_n = df[df.name==name]
        for cmdb_id in list(df_n.cmdb_id.unique()):
            df_nc = df_n[df_n.cmdb_id == cmdb_id]
            if np.mean(df_nc['value'].values) == 0 or np.mean(get_past(df_nc, curr_time, 60*12).value.values) == 0:
                print("Zero data for ", name, cmdb_id, ' skipping')
                df_thresh = df_thresh.append({'key': key, 'name':name, 'host': cmdb_id, 'thresh': -1}, ignore_index=True)
            else:
                df_nc = get_past(df_nc, curr_time, 60*12)
                df_nc = df_nc.set_index('timestamp')['value']

                od.infer_threshold(df_nc.values, threshold_perc=99.9)
                thresh = od.threshold
                print("Threshold for ", name, cmdb_id, ' is ', thresh)
                df_thresh = df_thresh.append({'key': key, 'name':name, 'host': cmdb_id, 'thresh': thresh}, ignore_index=True)
        df_thresh.to_csv('thresh.csv',index=False)