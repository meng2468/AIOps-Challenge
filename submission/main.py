'''
Example for data consuming.
'''
import requests
import json
import time
from collections import deque

from kafka import KafkaConsumer

from server_config import SERVER_CONFIGURATION
from lib.utils.data_types import PlatformIndex, BusinessIndex, Trace

from lib.utils import trace_build

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
    r = requests.post(SERVER_CONFIGURATION["SUBMIT_IP"], data=json.dumps(data))


def main():
    '''Consume data and react'''
    # Check authorities
    assert AVAILABLE_TOPICS <= CONSUMER.topics(), 'Please contact admin'

    data_tables = {
        'esb'   : deque(),
        'kpi'   : deque(),
        'trace' : deque(),
    }

    esb_time_window =   5 * 60 * 1000
    trace_time_window = 5 * 60 * 1000
    kpi_time_window =  60 * 60 * 1000

    def clean_tables():
        
        print(f"[DEBUG] Before cleanup sizes are: {len(data_tables['esb'])},{len(data_tables['kpi'])}, {len(data_tables['trace'])}")

        while data_tables['esb'][0].start_time < data_tables['esb'][-1].start_time - esb_time_window:
            data_tables['esb'].popleft() 

        while data_tables['kpi'][0].timestamp < data_tables['kpi'][-1].timestamp - kpi_time_window:
            data_tables['kpi'].popleft()

        while data_tables['trace'][0].start_time < data_tables['trace'][-1].start_time - trace_time_window:
            data_tables['trace'].popleft()

        print(f"[DEBUG] After cleanup, sizes are: {len(data_tables['esb'])},{len(data_tables['kpi'])}, {len(data_tables['trace'])}")

    for message in CONSUMER:
        data = json.loads(message.value.decode('utf8'))

        if message.topic == 'platform-index':
            data_tables['kpi'].extend(PlatformIndex(item) for stack in data['body'] for item in data['body'][stack])
        elif message.topic == 'business-index':
            data_tables['esb'].extend(BusinessIndex(item) for key in data['body'] for item in data['body'][key])

            clean_tables()

            start = time.time()
            trace_build.parse(data_tables['trace'])
            end = time.time()
            print(f'Parsing traces took {round(end - start,5)}s')

        else:  # message.topic == 'trace'
            data_tables['trace'].append(Trace(data))

if __name__ == '__main__':
    main()