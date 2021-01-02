'''
Example for data consuming.
'''
import requests
import json
import time
from collections import deque
import threading
import pickle
import sys
import csv

from kafka import KafkaConsumer

from server_config import SERVER_CONFIGURATION
from lib.utils.data_types import PlatformIndex, BusinessIndex, Trace

from lib.utils import trace

# Three topics are available: platform-index, business-index, trace.
# Subscribe at least one of them.
AVAILABLE_TOPICS = set(['platform-index', 'business-index', 'trace'])
CONSUMER = KafkaConsumer('platform-index', 'business-index', 'trace',
                         bootstrap_servers=[SERVER_CONFIGURATION["KAFKA_QUEUE"], ],
                         auto_offset_reset='latest',
                         enable_auto_commit=False,
                         security_protocol='PLAINTEXT')

def submit(ctx):
    '''Submit answer into stdout'''
    # print(json.dumps(data))
    assert (isinstance(ctx, list))
    for tp in ctx:
        assert(isinstance(tp, list))
        assert(len(tp) == 2)
        assert(isinstance(tp[0], str))
        assert(isinstance(tp[1], str) or (tp[1] is None))
    data = {'content': json.dumps(ctx)}
    if SERVER_CONFIGURATION['SUBMIT_IP']:
        print('Submitting...')
        r = requests.post(SERVER_CONFIGURATION["SUBMIT_IP"], data=json.dumps(data))

data = {
    'esb'   : deque(),
    'kpi'   : deque(),
    'trace' : deque(),
}

data_lock = threading.Lock()

if len(sys.argv) < 2:
    QUANTILES_PATH = './lib/models/quantiles.pickle'
else:
    print(f'Loading quantiles {sys.argv[1]}')
    QUANTILES_PATH = sys.argv[1]

with open(QUANTILES_PATH, 'rb') as f:
        QUANTILES = pickle.load(f)

last_submission = None
sub_lock = threading.Lock()

def process(new_data):
    ESB_TIME_WINDOW =   5 * 60 * 1000
    TRACE_TIME_WINDOW = 1 * 60 * 1000
    KPI_TIME_WINDOW =  60 * 60 * 1000

    def clean_tables(data_tables):
            print(f"[DEBUG] Before cleanup sizes are: {len(data_tables['esb'])},{len(data_tables['kpi'])}, {len(data_tables['trace'])}")

            while data_tables['esb'] and data_tables['esb'][0].start_time < data_tables['esb'][-1].start_time - ESB_TIME_WINDOW:
                data_tables['esb'].popleft() 

            while data_tables['kpi'] and data_tables['kpi'][0].timestamp < data_tables['kpi'][-1].timestamp - KPI_TIME_WINDOW:
                data_tables['kpi'].popleft()

            while data_tables['trace'] and data_tables['trace'][0].start_time < data_tables['trace'][-1].start_time - TRACE_TIME_WINDOW:
                data_tables['trace'].popleft()

            print(f"[DEBUG] After cleanup, sizes are: {len(data_tables['esb'])},{len(data_tables['kpi'])}, {len(data_tables['trace'])}")

    global last_submission

    # update global data
    with data_lock:
        data['esb'].extend(new_data['esb'])
        data['kpi'].extend(new_data['kpi'])
        data['trace'].extend(new_data['trace'])
        clean_tables(data)
        if data['trace']:
            now = time.time()
            
            # check if can submit (5min window submission)
            if not last_submission or now - last_submission >= 5*60: 
                result = trace.table(QUANTILES, data['trace'], debug=False)
                if result:
                    with open('anomalies_found.csv','a+') as f:
                        writer = csv.writer(f)
                        writer.writerow([now, *result])
                    print(result)
                    submit(result)
                    last_submission = now


    

def main():
    '''Consume data and react'''
    # Check authorities
    assert AVAILABLE_TOPICS <= CONSUMER.topics(), 'Please contact admin'

    data_tables = {
        'esb'   : deque(),
        'kpi'   : deque(),
        'trace' : deque(),
    }

    for message in CONSUMER:
        data = json.loads(message.value.decode('utf8'))

        if message.topic == 'platform-index':
            data_tables['kpi'].extend(PlatformIndex(item) for stack in data['body'] for item in data['body'][stack])
        elif message.topic == 'business-index':
            data_tables['esb'].extend(BusinessIndex(item) for key in data['body'] for item in data['body'][key])

            threading.Thread(target=process, args=(data_tables,)).start()
            data_tables = {
                'esb'   : deque(),
                'kpi'   : deque(),
                'trace' : deque(),
            }

        else:  # message.topic == 'trace'
            data_tables['trace'].append(Trace(data))

if __name__ == '__main__':
    main()