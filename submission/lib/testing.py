import numpy as np
import pandas as pd
import pickle

from utils.data_types import Trace
import utils.trace as trace

data_folder = '../../data/full_data'
def load_anomalies():
    anomalies = pd.read_csv(data_folder + '/anomaly_list.csv')
    return anomalies

def format_trace(anomaly):
    trace_df = pd.read_csv(data_folder + '/' + anomaly['folder'] + '/trace.csv')
    max_time = np.max(trace_df['startTime'].values)
    start_time = max_time - 10*60*1000
    trace_df = trace_df[trace_df.startTime >= 10*60*1000].sort_values('startTime')

    traces = []
    for i in range(len(trace_df)):
        traces.append(Trace(trace_df.iloc[i, :]))
    
    return traces

if __name__ == '__main__':
    pickle_path = 'models/quantiles_0.0005.pickle'
    df = pd.DataFrame(columns=['fault'])
    with open(pickle_path, 'rb') as f:
        limits = pickle.load(f)
    anomalies = load_anomalies()

    for i in range(len(anomalies)):
        anomaly = anomalies.iloc[i,:]
        print('Loading anomaly', anomaly['folder'])
        traces = format_trace(anomalies.iloc[i,:])
        print('Run meow meow')
        result = trace.get_anomalous_hosts_count(limits,traces)
        result['fault'] = anomaly['fault']
        result['host'] = anomaly['host']
        result['kpi'] = anomaly['kpi']
        df = df.append(result, ignore_index=True)
        df.to_csv('results.csv', index=False)