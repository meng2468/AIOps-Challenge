from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from server_config import SERVER_CONFIGURATION
import time
import json
import pandas as pd
import os
import csv
import sys

if __name__ == '__main__':
    DATA_PATH = os.path.join('.')

    try:
        admin_client = KafkaAdminClient(
            bootstrap_servers=SERVER_CONFIGURATION['KAFKA_QUEUE'], 
            client_id='test'
        )
        AVAILABLE_TOPICS = set(['platform-index', 'business-index', 'trace'])

        topic_list = [NewTopic(name=name, num_partitions=1, replication_factor=1) for name in AVAILABLE_TOPICS]
        admin_client.create_topics(new_topics=topic_list, validate_only=False)
    except:
        print('Topics already created, proceed')

    producer = KafkaProducer(bootstrap_servers=SERVER_CONFIGURATION['KAFKA_QUEUE'])

    print('Loading data...')

    if len(sys.argv) < 2:
        print('Loading default test data')
        DIR = 'server_data/'
    else:
        print(f'Loading {sys.argv[1]} data')
        DIR = sys.argv[1]

    # Load data here
    esb = pd.read_csv(f'{DIR}/esb.csv')
    kpi = pd.read_csv(f'{DIR}/host.csv')

    TRACE_BATCH_SIZE = 10000
    trace_fd = pd.read_csv(f'{DIR}/trace.csv', chunksize=TRACE_BATCH_SIZE)
    trace = next(trace_fd)

    esb_index = 0
    kpi_index = 0
    trace_index = 0

    max_esb = 720
    max_kpi = 1145892
    max_trace = 19741832

    print('Data loading completed. Initiating message sequence.')
    print(f'Total sizes to be read: ({max_esb}, {max_kpi}, {max_trace})')
    counter = 0
    while esb_index < max_esb or kpi_index < max_kpi or trace_index + counter * TRACE_BATCH_SIZE < max_trace:
        esb_time = esb.iloc[esb_index].startTime
        kpi_time = kpi.iloc[kpi_index].timestamp
        trace_time = trace.iloc[trace_index].startTime
        v = min(esb_time, kpi_time, trace_time)
        
        # TODO if required, we can include here the real-time logic to wait before sending

        if esb_time == v:
            data = {
                'body' : {'esb': [json.loads(esb.iloc[esb_index].to_json())]},
                'startTime' : time.time()
            }
            producer.send('business-index', json.dumps(data).encode('utf-8'))
            esb_index+=1
            print(f'ESB data sent! ({esb_index}, {kpi_index}, {trace_index + counter * TRACE_BATCH_SIZE})')
        elif kpi_time == v:
            data = {
                'body' : {
                    'key': [json.loads(kpi.iloc[kpi_index].to_json())] # key doesnt matter
                },
                'timestamp' : time.time() 
            }
            producer.send('platform-index', json.dumps(data).encode('utf-8'))
            kpi_index += 1
        else:
            data = json.loads(trace.iloc[trace_index].to_json())
            
            producer.send('trace', json.dumps(data).encode('utf-8'))
            trace_index += 1
            if trace_index % TRACE_BATCH_SIZE == 0:
                trace = next(trace_fd)
                trace_index = 0
                counter += 1
                max_trace -= TRACE_BATCH_SIZE