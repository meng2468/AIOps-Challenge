import pandas as pd
import numpy as np
import zipfile

from datetime import datetime
from datetime import timedelta
import time

import csv
import os

def unzip_all():
    for file in os.listdir('.'):
        if file[-3:] == 'zip':
            zip = zipfile.ZipFile(file)
            zip.extractall()

def get_timestamp_window(str_time):
    start_time = datetime.strptime(str_time,'%Y/%m/%d %H:%M')
    # Change hours to time difference between you and China in April/May
    window_start = start_time - timedelta(hours=6) - timedelta(minutes=25)
    window_finish = start_time - timedelta(hours=6) + timedelta(minutes=5)
    return window_start.timestamp()*1000, window_finish.timestamp()*1000

def get_formatted_anoms(anomalies):
    for i in range(len(anomalies)):
        anomaly = {}
        anomaly['folder'] = str(i) + '_' + anomalies['fault_desrcibtion'].values[i].replace(' ', '_')
        anomaly['fault'] = anomalies['fault_desrcibtion'].values[i]
        anomaly['host'] = anomalies['name'].values[i]
        anomaly['container'] = anomalies['container'].values[i]
        anomaly['kpi'] = anomalies['kpi'].values[i]

        if anomalies['start_time'].isnull()[i]:
            anomaly['chinese_start'] = anomalies['log_time'].values[i]
            anomaly['wt_start'], anomaly['wt_stop'] = get_timestamp_window(anomalies['log_time'].values[i])
        else:
            anomaly['chinese_start'] = anomalies['start_time'].values[i]
            anomaly['wt_start'], anomaly['wt_stop'] = get_timestamp_window(anomalies['start_time'].values[i])
 
        if i == 0:
            df = pd.DataFrame(anomaly, index=[0])
        else:
            df = df.append(anomaly, ignore_index=True)
    return df

def load_esb(path, start, stop):
    for file in os.listdir(path):
        df = pd.read_csv(path + '/' + file)
        print('Loading', len(df), 'rows from', file)
        df = df[(df.startTime >= start) & (df.startTime <= stop)]
        print('Returning', len(df), 'relevant rows')
    return df

def load_trace(path, start, stop):
    dfs = []
    for file in os.listdir(path):
        df = pd.read_csv(path+'/'+file)
        print('Loading', len(df), 'rows from', file)
        df = df[(df.startTime >= start) & (df.startTime <= stop)] 
        print('Returning', len(df), 'relevant rows')
        dfs.append(df)
    return pd.concat(dfs)

def load_kpi(path, start, stop):
    dfs = []
    for file in os.listdir(path):
        df = pd.read_csv(path+'/'+file)
        print('Loading', len(df), 'rows from', file)
        df = df[(df.timestamp >= start) & (df.timestamp <= stop)] 
        print('Returning', len(df), 'relevant rows')
        dfs.append(df)
    return pd.concat(dfs)

def generate_folder(anomaly):
    os.makedirs(anomaly['folder'])
    data_folder = datetime.strptime(anomaly['chinese_start'], '%Y/%m/%d %H:%M').strftime('%Y_%m_%d')

    print('Loading data')
    esb = load_esb(data_folder+'/业务指标', anomaly['wt_start'], anomaly['wt_stop'])
    trace = load_trace(data_folder+'/调用链指标', anomaly['wt_start'], anomaly['wt_stop'])
    kpi = load_kpi(data_folder+'/平台指标', anomaly['wt_start'], anomaly['wt_stop'])

    print('Saving data')
    esb.to_csv(anomaly['folder'] + '/esb.csv', index=False)
    trace.to_csv(anomaly['folder'] + '/trace.csv', index=False)
    kpi.to_csv(anomaly['folder'] + '/host.csv', index=False)

if __name__ == '__main__':
    unzip_all()
    anomalies = pd.read_csv('anomalies.data')
    anomalies = get_formatted_anoms(anomalies)
    for i in range(len(anomalies)):
        print('*'*60)
        print('Extracting anomaly data for', anomalies['folder'].values[i])
        generate_folder(anomalies.iloc[i,:])
    anomalies.to_csv('anomaly_list.csv', index=False)
