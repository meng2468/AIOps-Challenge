import numpy as np
import pandas as pd
import os

import preprocessing as pp
import model_gen as vae

from tensorflow import keras
from tensorflow.keras import layers

time_step = 144

thresh_dir = ''
model_dir = ''


def find_anom(host, dfs):
    problems = []
    key = ''
    for k in dfs:
        if host in dfs[k].cmdb_id.unique():
            key = k
    if key == '':
        print('Host not found!')
        return
    df = dfs[key]
    thresh = pd.read_csv(thresh_dir+'thresh_'+str(time_step)+'.csv')

    df_h = df[df.cmdb_id==host]
    for name in df_h['name'].unique():
        df_nh = df_h[df_h.name == name]

        #Pre-process data  
        x_test = pp.get_host_kpi_data(df_nh, time_step)
        if key+'_'+name in os.listdir(model_dir):
            print('-'*40)
            print("Running detection for ", host, name)
            if x_test == []:
                print(name, host, 'not enough data')
                continue

            model = keras.models.load_model(model_dir+key+'_'+name)
            x_test_pred = model.predict(x_test)
            test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
            test_mae_loss = test_mae_loss.reshape((-1))
            anomalies = np.greater(test_mae_loss, thresh[thresh.KPI==key+'_'+name]['thresh'].values[0])

            if True in anomalies:
                print("Anomaly in ", host, name)
                problems.append((host, name))
        else:
            print(name, host, ' model does not exist')
    if problems == []:
        print("No anomalies found in ", host)
    return problems