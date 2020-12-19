'''
Example for data consuming.
'''
import json
import os

import pandas as pd
import numpy as np
import requests
from kafka import KafkaConsumer
import time
import csv
import threading

# from models.esb_detection.vae import detect as esb_vae
from models.esb_detection.seas_decomp import detect as esb_seas
from models.esb_detection.heuristic import detect as esb_heur
import trace_detector
# from models.kpi_detection.vae import detect as kpi_detection
from server_config import SERVER_CONFIGURATION

# Three topics are available: platform-index, business-index, trace.
# Subscribe at least one of them.
AVAILABLE_TOPICS = set(['platform-index', 'business-index', 'trace'])
CONSUMER = KafkaConsumer('platform-index', 'business-index', 'trace',
                         bootstrap_servers=[SERVER_CONFIGURATION["KAFKA_QUEUE"], ],
                         auto_offset_reset='latest',
                         enable_auto_commit=False,
                         security_protocol='PLAINTEXT')

def print_sep():
    print('*'*60)

class PlatformIndex():  # pylint: disable=too-few-public-methods
    '''Structure for platform indices'''


    def __init__(self, data):
        self.item_id = [data['itemid']]
        self.name = [data['name']]
        self.bomc_id = [data['bomc_id']]
        self.timestamp = [data['timestamp']]
        self.value = [data['value']]
        self.cmdb_id = [data['cmdb_id']]

    def to_dataframe(self):
        return pd.DataFrame.from_dict(self.__dict__)

class BusinessIndex():  # pylint: disable=too-few-public-methods
    '''Structure for business indices'''


    def __init__(self, data):
        self.serviceName = [data['serviceName']]
        self.startTime = [data['startTime']]
        self.avg_time = [data['avg_time']]
        self.num = [data['num']]
        self.succee_num = [data['succee_num']]
        self.succee_rate = [data['succee_rate']]

    def to_dataframe(self):
        return pd.DataFrame.from_dict(self.__dict__)

class Trace():  # pylint: disable=invalid-name,too-many-instance-attributes,too-few-public-methods
    '''Structure for traces'''


    def __init__(self, data):
        self.callType = [data['callType']]
        self.startTime = [data['startTime']]
        self.elapsedTime = [data['elapsedTime']]
        self.success = [data['success']]
        self.traceId = [data['traceId']]
        self.id = [data['id']]
        self.pid = [data['pid']]
        self.cmdb_id = [data['cmdb_id']]

        if 'serviceName' in data:
        #     # For data['callType']
        #     #  in ['CSF', 'OSB', 'RemoteProcess', 'FlyRemote', 'LOCAL']
            self.serviceName = [data['serviceName']]

        if data['callType'] in ['LOCAL','JDBC'] and 'dsName' in data:
            self.serviceName = [data['dsName']]

        elif data['callType'] == 'OSB' or data['callType'] == 'RemoteProcess' and 'cmdb_id' in data:
            self.serviceName = [data['cmdb_id']]
        
        # if 'dsName' in data and data['callType'] == 'JDBC':
        #     # For data['callType'] in ['JDBC', 'LOCAL']
        #     self.serviceName = [data['dsName']]

    def to_dataframe(self):
        return pd.DataFrame.from_dict(self.__dict__)

ANOMALY_FILENAME = 'anomalies_list.csv'
def submit(ctx, timestamp):
    global ANOMALY_FILENAME

    if not ctx:
        return

    with open(ANOMALY_FILENAME, 'a') as f:
        print(f'[INFO][{timestamp}] Writing new result to file: {ctx}')
        writer = csv.writer(f)
        writer.writerow([timestamp, time.time(), *ctx])

    '''Submit answer into stdout'''
    # print(json.dumps(data))
    assert (isinstance(ctx, list))
    for tp in ctx:
        assert(isinstance(tp, list))
        assert(len(tp) == 2)
        assert(isinstance(tp[0], str))
        assert(isinstance(tp[1], str) or (tp[1] is None))
    data = {'content': json.dumps(ctx)}
    r = requests.post(SERVER_CONFIGURATION["SUBMIT_IP"], data=json.dumps(data))

last_anomaly_timestamp = None
timestamp_lock = threading.Lock()
def handle_anomaly(df, timestamp):
    traces = df['trace']
    kpis = df['kpi']
    print(traces)
    anomalous_hosts = trace_detector.detect(traces, kpis)
    
    print(anomalous_hosts)    
    submit(anomalous_hosts, timestamp) 

df = {
        'esb': pd.DataFrame(columns=['serviceName','startTime','avg_time','num','succee_num','succee_rate']), 
        'trace': pd.DataFrame(columns=['startTime','elapsedTime','success','traceId','id','pid','cmdb_id','serviceName','callType']), 
        'kpi': pd.DataFrame(columns=['item_id','name','bomc_id','timestamp','value','cmdb_id'])
    }

df_lock = threading.Lock()

def analyzer(esb_array, trace_array, kpi_array):
    esb = pd.concat(map(lambda esb: esb.to_dataframe(), esb_array))

    try:
        trace = pd.concat(map(lambda trace: trace.to_dataframe(), trace_array))
    except ValueError:
        print('No trace data available. Using empty dataframe...')
        trace = pd.DataFrame()

    try:
        kpi = pd.concat(map(lambda kpi: kpi.to_dataframe(), kpi_array))
    except:
        print('No kpi data available. Using empty dataframe...')
        kpi = pd.DataFrame()

    with df_lock:
        df['esb'] = pd.concat([df['esb'], esb], ignore_index=True).sort_values('startTime')
        df['trace'] = pd.concat([df['trace'], trace], ignore_index=True)
        df['kpi'] = pd.concat([df['kpi'], kpi], ignore_index=True)
    

    window_size_ms = 1000 * 60 * 10 # window size: 5min
    timestamp = int(df['esb'].iloc[-1]['startTime'])

    # remove from all dfs
    for key in df:
        dataframe = df[key]
        print(f'[DEBUG] Size of dataframe {key} was {len(df[key])}')
        if (key == 'kpi'):
            df[key] = dataframe[dataframe['timestamp'] >= timestamp - window_size_ms*6]
        else:
            df[key] = dataframe[dataframe['startTime'] >= timestamp - window_size_ms]
        print(f'[DEBUG] Size of dataframe {key} is now {len(df[key])}')

          

    esb_is_anomalous = esb_heur.find_anom(df['esb'])

    if(esb_is_anomalous):
        print('ESB anomaly detected')
        global last_anomaly_timestamp
        with timestamp_lock:
            if last_anomaly_timestamp is None:
                last_anomaly_timestamp = timestamp
                print(f'[INFO] Timestamp was None, assign {timestamp}')
            elif timestamp - last_anomaly_timestamp < 1000*60*10: # Last anomaly was detected less than 5 min ago
                print(f'[INFO] Last anomaly was detected less than 5 min ago, skipping... {timestamp}')
                return 
            else: # Last anomaly was detected more than 5 min ago
                print(f'[INFO] New timestamp, assign {timestamp}')
                last_anomaly_timestamp = timestamp
        print('Started 3min timer...')
        time.sleep(61*3)
        print('Starting trace_detector thread')
        t = threading.Thread(target=handle_anomaly, args=(df,timestamp))
        t.start()
        # TODO do we need to wait for it?
    else:
        print('No ESB anomaly')

    del esb
    del trace
    del kpi

def main():
    '''Consume data and react'''
    # Check authorities
    assert AVAILABLE_TOPICS <= CONSUMER.topics(), 'Please contact admin'

    esb_data = []
    trace_data = []
    kpi_data = []

    i = 0
    print(f'Starting connection with server at {SERVER_CONFIGURATION["KAFKA_QUEUE"]}')
    for message in CONSUMER:
        i += 1
        data = json.loads(message.value.decode('utf8'))
        if message.topic == 'platform-index':
            extra_kpis = [PlatformIndex(item) for stack in data['body'] for item in data['body'][stack]]
            kpi_data.extend(extra_kpis)

        elif message.topic == 'business-index':
            new_business_data = [BusinessIndex(item) for key in data['body'] for item in data['body'][key]]
            esb_data.extend(new_business_data)

            # Start new thread to analyze
            threading.Thread(target=analyzer, args=(esb_data, trace_data, kpi_data)).start()

            # Empty global ones 
            esb_data = []
            trace_data = []
            kpi_data = []

        else:  # message.topic == 'trace'
            trace_data.append(Trace(data))



if __name__ == '__main__':
    # Initialize the MicroRCA detector
    main()
