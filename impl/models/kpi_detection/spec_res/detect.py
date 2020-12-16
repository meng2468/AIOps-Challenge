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

def get_thresh(host):
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
        print('Checking', host)
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
            df_hk = df[df.cmdb_id == host][df.name == kpi]
            
            if len(df_hk) == 0:
                print('No valid data to use, skipping')
                continue
                
            data = get_past(df_hk, np.max(df_hk.timestamp.unique()), 30)['value']
            if thresh == -1:
                if sum(data[-10:-5]) > 0:
                    print("Non-Zero Anomaly!")
                    anoms.append((host, kpi))
            elif np.isnan(thresh):
                print('NaN, Skipping!')
            else:
                outliers = od.predict(data.values)['data']
                if np.sum(outliers['is_outlier'][-10:-5]) > 0:
                    print(outliers['is_outlier'])
                    print("ST Threshold Anomaly!")
                    anoms.append((host,kpi))
        print(host, ' completed in ', time.time() - start_host)
    print('Completed detection of ', len(hosts), 'hosts in ', time.time() - start, 'seconds')
    return anoms



    
    