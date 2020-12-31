import pandas as pd
import numpy as np
import os

import time
from alibi_detect.od import SpectralResidual
from datetime import datetime



def run_gen(perc):
    data_path = '/Users/baconbaker/Documents/Studium/ANM/anm-project/data/train_data/host'
    dfs = {}
    for path in os.listdir(data_path):
        dfs[path[:-4]] = pd.read_csv(data_path+'/'+path)

    df_info = pd.read_csv('kpi_summary_info.data')

    window_size = 10
    od = SpectralResidual(
        window_amp=window_size,
        window_local=window_size,
        n_est_points=5,
        n_grad_points=5
    )

    per1_kpis = df_info[(df_info.interval=='1min') & (df_info.is_flat == False)]['kpi'].unique()
    per5_kpis = df_info[(df_info.interval=='5min') & ((df_info.is_flat == False))]['kpi'].unique()

    df_thresh = pd.DataFrame(columns=['name','host','thresh'])

    for df_name in dfs:
        print('*'*50)
        print('Running generation for', df_name)
        interval = 0
        start_key = time.time()

        df = dfs[df_name]
        kpis = dict(tuple(df.groupby(['cmdb_id', 'name'])))
        res = {}

        for key in kpis:
            kpis[key]['timestamp'] = kpis[key]['timestamp'].apply(lambda x: datetime.fromtimestamp(x/1000.0))
            kpis[key] = kpis[key].set_index('timestamp').sort_index()

        print('Calculating rolling window')
        for key in kpis: 
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
            d = res[key].dropna()
            if len(res[key]) == 0:
                print("ITS EMPTY", key)
                continue
            od.infer_threshold(d, threshold_perc=perc)
            thresh = od.threshold
            df_thresh = df_thresh.append({'name':key[1], 'host': key[0], 'thresh': thresh}, ignore_index=True)

        df_thresh.to_csv('thresh_'+str(perc).replace('.','_')+'.data',index=False)

for perc in [99.7,99.9,99.99,99.999]:
    run_gen(perc)