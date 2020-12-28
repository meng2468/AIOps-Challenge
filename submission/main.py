'''
Example for data consuming.
'''
import requests
import json

from kafka import KafkaConsumer

from server_config import SERVER_CONFIGURATION

# Three topics are available: platform-index, business-index, trace.
# Subscribe at least one of them.
AVAILABLE_TOPICS = set(['platform-index', 'business-index', 'trace'])
CONSUMER = KafkaConsumer('platform-index', 'business-index', 'trace',
                         bootstrap_servers=[SERVER_CONFIGURATION["KAFKA_QUEUE"], ],
                         auto_offset_reset='latest',
                         enable_auto_commit=False,
                         security_protocol='PLAINTEXT')


class PlatformIndex():  # pylint: disable=too-few-public-methods
    '''Structure for platform indices'''

    __slots__ = ['item_id', 'name', 'bomc_id', 'timestamp', 'value', 'cmdb_id']

    def __init__(self, data):
        self.item_id = data['itemid']
        self.name = data['name']
        self.bomc_id = data['bomc_id']
        self.timestamp = data['timestamp'] 
        self.value = data['value']
        self.cmdb_id = data['cmdb_id']


class BusinessIndex():  # pylint: disable=too-few-public-methods
    '''Structure for business indices'''

    __slots__ = ['service_name', 'start_time', 'avg_time', 'num',
                 'succee_num', 'succee_rate']

    def __init__(self, data):
        self.service_name = data['serviceName']
        self.start_time = data['startTime']
        self.avg_time = data['avg_time']
        self.num = data['num']
        self.succee_num = data['succee_num']
        self.succee_rate = data['succee_rate']


class Trace():  # pylint: disable=invalid-name,too-many-instance-attributes,too-few-public-methods
    '''Structure for traces'''

    __slots__ = ['call_type', 'start_time', 'elapsed_time', 'success',
                 'trace_id', 'id', 'pid', 'cmdb_id', 'service_name', 'ds_name']

    def __init__(self, data):
        self.call_type = data['callType']
        self.start_time = data['startTime']
        self.elapsed_time = data['elapsedTime']
        self.success = data['success']
        self.trace_id = data['traceId']
        self.id = data['id']
        self.pid = data['pid']
        self.cmdb_id = data['cmdb_id']

        if 'serviceName' in data:
            # For data['callType']
            #  in ['CSF', 'OSB', 'RemoteProcess', 'FlyRemote', 'LOCAL']
            self.service_name = data['serviceName']
        if 'dsName' in data:
            # For data['callType'] in ['JDBC', 'LOCAL']
            self.ds_name = data['dsName']


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
        'esb'   : [],
        'kpi'   : [],
        'trace' : [],
    }

    for message in CONSUMER:
        data = json.loads(message.value.decode('utf8'))
        if message.topic == 'platform-index':
            data_tables['kpi'].extend([PlatformIndex(item) for item in data['body'][stack]] for stack in data['body'])
        elif message.topic == 'business-index':
            data_tables['esb'].extend([BusinessIndex(item) for item in data['body'][key]] for key in data['body'])
        else:  # message.topic == 'trace'
            data_tables['trace'].append(Trace(data))
            


if __name__ == '__main__':
    main()