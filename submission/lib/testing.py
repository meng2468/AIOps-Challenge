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

def save_results(pickle_path, output):
    df = pd.DataFrame(columns=['fault'])
    with open(pickle_path, 'rb') as f:
        limits = pickle.load(f)
    anomalies = load_anomalies()

    for i in range(len(anomalies)):
        anomaly = anomalies.iloc[i,:]
        print('Loading anomaly', anomaly['folder'])
        traces = format_trace(anomalies.iloc[i,:])
        print('Run meow meow')
        _, result = trace.get_anomalous_hosts_count(limits,traces)
        for i in result:
            if type(result[i]) == list:
                result[i] = int((result[i][0] / result[i][1])*10000)/10000
        result['fault'] = anomaly['fault']
        result['host'] = anomaly['host']
        result['kpi'] = anomaly['kpi']
        df = df.append(result, ignore_index=True)
        df.to_csv(output, index=False)

# Table anomaly logic
def get_anomaly(anom_num, results):
    anom = ''
    highest = 0
    for column in results.columns[6:-3]:
        if results.iloc[anom_num,:][column] > highest:
            highest = results.iloc[anom_num,:][column]
            anom = column
    return anom

def check_max_anom(results):
    anoms_checked = 0
    anoms_found = 0
    print('*'*40)
    print('Testing host anomaly localisation')
    for i in range(len(results)):
        anoms_checked += 1
        if results.iloc[i,:]['host'] == get_anomaly(i, results):
            anoms_found += 1
            print(results.iloc[i,:]['fault'], 'detected')
        else:
            print(results.iloc[i,:]['fault'], 'not-detected')
    print('Accuracy: ', anoms_found/anoms_checked)


# test different window sizes
# test different thresholds
if __name__ == '__main__':
    for i in range(5):
        pickle_path = 'models/quantiles_0.000'+str(i)+'.pickle'
        save_results(pickle_path,'results'+str(i)+'.csv')
