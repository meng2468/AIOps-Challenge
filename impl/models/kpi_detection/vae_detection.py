import numpy as np
import pandas as pd
import os

from tensorflow import keras
from sklearn import preprocessing
import time

data_path='/content/host'
model_dir = '/content/drive/MyDrive/anm-data/models/' # Model save-file directory

dfs={}
for file in os.listdir(data_path):
    print('Saving ' + file[:-4] + ' into dfs')
    dfs[file[:-4]] = pd.read_csv(data_path+'/'+file)

#Reformatting
dfs['db_oracle_11g'] = dfs.pop('db')
dfs['dcos_container'] = dfs.pop('container')
dfs['mw_redis'] = dfs.pop('redis')
dfs['os_linux'] = dfs.pop('os')
dfs['dcos_docker'] = dfs.pop('docker')

#Preprocessing
def normalise(df):
    df.loc[:,['value']] = preprocessing.scale(df['value'].values)
    return df

def gen_train_seq(values, time_steps=288):
    output = []
    for i in range(len(values) - time_steps):
        output.append(values[i : (i + time_steps)])
        
    return np.stack(output)