import pandas as pd
import numpy as np

import time
import os
from datetime import datetime

from alibi_detect.od import SpectralResidual

def find_anoms(hosts, df):
    start = time.time()
    kpis = dict(tuple(df.groupby(['cmdb_id', 'name'])))
    res = {}
    anoms = []
    
    df_info = pd.read_csv('kpi_summary_info.data')
    df_thresh = pd.read_csv('thresh_99_999.data')

    for key in kpis:
        kpis[key]['timestamp'] = kpis[key]['timestamp'].apply(lambda x: datetime.fromtimestamp(x/1000.0))
        kpis[key] = kpis[key].set_index('timestamp').sort_index()

    per1_kpis = df_info[(df_info.interval=='1min') & (df_info.is_flat == False)]['kpi'].unique()
    per5_kpis = df_info[(df_info.interval=='5min') & ((df_info.is_flat == False))]['kpi'].unique()

    print('Calculating rolling window')
    for key in kpis: 
        if key[0] in hosts:
            if kpis[key]['value'].std() == 0:
                continue
            elif key[1] in per1_kpis:
                d = kpis[key]['value'].resample('T').mean().interpolate()
            elif key[1] in per5_kpis:
                d = kpis[key]['value'].resample('5T').mean().interpolate()
            else:
                continue
            d = (d - d.mean())/d.std()
            res[key] = d.rolling(10).mean()


    for key in res:
        print('Determining threshold for', key)
        if len(df_thresh[(df_thresh.host == key[0]) & (df_thresh.name==key[1])]) == 0:
            print('Anomaly,  std in train was 0, now its not')
            anoms.append((key[1],key[0]))
            continue
        thresh = df_thresh[(df_thresh.host == key[0]) & (df_thresh.name==key[1])]['thresh'].values[0]
        if np.isnan(thresh):
            print("SR didn't generate threshhold because of low std for window > 10, skipping")
            continue

        d = res[key].dropna()
        od = SpectralResidual(
                threshold=thresh,
                window_amp=10,
                window_local=10,
                n_est_points=5,
                n_grad_points=5
            )
        if len(d) < 10:
            print('Rolling window data empty! Skipping')
            continue
        outliers = od.predict(d.values)['data']
        if np.sum(np.sum(outliers['is_outlier'][-5:-2])) > 0:
            print(outliers['is_outlier'])
            print("ST Threshold Anomaly!")
            anoms.append((key[1],key[0]))
    print("It took", time.time()-start, "seconds to find", len(anoms),"anomalies")
    return anoms

print(find_anoms(pd.read_csv('/Users/baconbaker/Documents/Studium/ANM/anm-project/data/tiago_tests/test9/kpis.csv').cmdb_id.unique(), pd.read_csv('/Users/baconbaker/Documents/Studium/ANM/anm-project/data/tiago_tests/test9/kpis.csv')))