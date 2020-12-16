import pandas as pd
import numpy as np

import time
import os

from alibi_detect.od import SpectralResidual

def load_test():
    data_path = '../../../../data/test_data/host'
    df = pd.DataFrame(['item_id','name','bomc_id','timestamp','value','cmdb_id'])
    for file in os.listdir(data_path):
        df = pd.concat([df, pd.read_csv(data_path+'/'+file)], ignore_index=True) 
    return df

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

def get_key_thresh(host):
    df_thresh = pd.read_csv('thresh.data')
    df_thresh = df_thresh[df_thresh.host==host]

    thresh = df_thresh[['name', 'thresh']].set_index('name')
    thresh = thresh.to_dict()['thresh']
    return thresh

def find_anoms(hosts, df):
    start = time.time()
    anoms = []
    for host in hosts:
        start_host = time.time()
        print('*'*60)
        print('Checking ', host)
        thresholds = get_key_thresh(host)
        for kpi in thresholds:
            thresh = thresholds[kpi]
            print('Threshold ', thresh, kpi)
            od = SpectralResidual(
                threshold=thresh,
                window_amp=30,
                window_local=30,
                n_est_points=5,
                n_grad_points=5
            )
            df_hk = df[(df.cmdb_id == host) & (df.name == kpi)]
            
            if len(df_hk) == 0:
                print('No valid data to use, skipping')
                continue

            data = get_past(df_hk, np.max(df_hk.timestamp.unique()), 30)['value']
            data = np.concatenate([data, data])
            
            if thresh == -1:
                if sum(data[-3:]) > 0:
                    print("Was expecting zero's, got something else, anomaly!")
                    anoms.append((host, kpi))
            elif np.isnan(thresh):
                if  np.std(data[-3:]) != 0:
                    print('Anomaly, std should have been zero!')
                    anoms.append((host, kpi))
            else:
                outliers = od.predict(data)['data']
                if np.mean(data) == 0 or np.std(data) == 0:
                    print("Mean is zero or std is zero")
                    anoms.append((host,kpi))
                if np.sum(outliers['is_outlier'][-27:-30]) > 0:
                    print("ST Threshold Anomaly!")
                    anoms.append((host,kpi))
        print(host, ' completed in ', time.time() - start_host)
    print('Completed detection of ', len(hosts), 'hosts in ', time.time() - start, 'seconds')
    return anoms

find_anoms(load_test().cmdb_id.unique(), load_test())
    