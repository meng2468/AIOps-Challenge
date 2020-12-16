import pandas as pd
import numpy as np

from alibi_detect.od import SpectralResidual

def get_past(df, curr_time, window_size):
    step_size = 1000*60
    df_new = pd.DataFrame(columns=['timestamp', 'value'])

    for time in [curr_time - step_size*x for x in range(window_size)][::-1]:
        if len(df[df.timestamp == time]) == 0:
            plug_time = np.max(df[df.timestamp < time]['timestamp'])
            row = {}
            row['timestamp'] = time
            row['value'] = df.loc[df.timestamp==plug_time, 'value'].values[0]
            df_new = df_new.append(row, ignore_index=True)
        else:
            row = {}
            row['timestamp'] = time
            row['value'] = df.loc[df.timestamp==time, 'value'].values[0]
            df_new = df_new.append(row, ignore_index=True)
    
    return df_new

def is_anom(df, thresh, anom_time):
    od = SpectralResidual(
        threshold=thresh,
        window_amp=30,
        window_local=30,
        n_est_points=5,
        n_grad_points=5
    )
    df = get_past(df, anom_time, 30)
    return bool(od.predict(df['value'].values)['data']['is_outlier'][-1])


def find_anom(host, dfs, anom_time):
    df_thresh = pd.read_csv('thresh.csv')
    key = ''
    for k in dfs:
        if host in dfs[k].cmdb_id.unique():
            key = k
    df = dfs[k][dfs[k].cmdb_id == host]]
    for name in df.name.unique():
        thresh = df_thresh[df_thresh.name == name][df_thresh.cmdb_id==host].value.values[0]
        print(thresh)

    
    