'''
Example for data consuming.
'''
import requests
import json

from kafka import KafkaConsumer

from server_config import SERVER_CONFIGURATION

import pandas as pd

from models.esb_detection.vae import detect as detect_esb

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
            # For data['callType']
            #  in ['CSF', 'OSB', 'RemoteProcess', 'FlyRemote', 'LOCAL']
            self.serviceName = [data['serviceName']]
        if 'dsName' in data and data['callType'] == 'JDBC':
            # For data['callType'] in ['JDBC', 'LOCAL']
            self.serviceName = [data['dsName']]

    def to_dataframe(self):
        return pd.DataFrame.from_dict(self.__dict__)

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

    #submit([['docker_003', 'container_cpu_used']])  FIXME Why was this here? 
    i = 0

    df = {
        'esb': pd.DataFrame(columns=['serviceName','startTime','avg_time','num','succee_num','succee_rate']), 
        'trace': pd.DataFrame(columns=['startTime','elapsedTime','success','traceId','id','pid'',cmdb_id','serviceName','callType']), 
        'kpi': pd.DataFrame(columns=['item_id','name','bomc_id','timestamp','value','cmdb_id'])
    }

    print(f'Starting connection with server at {SERVER_CONFIGURATION["KAFKA_QUEUE"]}')
    for message in CONSUMER:
        i += 1
        data = json.loads(message.value.decode('utf8'))
        if message.topic == 'platform-index':
            # data['body'].keys() is supposed to be
            # ['os_linux', 'db_oracle_11g', 'mw_redis', 'mw_activemq',
            #  'dcos_container', 'dcos_docker']
            new_df = pd.concat(map(lambda x: x.to_dataframe(), [PlatformIndex(item) for stack in data['body'] for item in data['body'][stack]]))
            df['kpi'] = pd.concat([df['kpi'],new_df], ignore_index=True)

        elif message.topic == 'business-index':
            # data['body'].keys() is supposed to be ['esb', ]
            new_df = pd.concat(map(lambda x: x.to_dataframe(), [BusinessIndex(item) for key in data['body'] for item in data['body'][key]]))
            df['esb'] = pd.concat([df['esb'],new_df], ignore_index=True)

            window_size_ms = 1000 * 60 * 5 # window size: 5min
            timestamp = int(new_df.iloc[-1]['startTime']) - window_size_ms # check the most recent
    
            # remove from all dfs
            for key in df:
                dataframe = df[key]
                df[key] = dataframe[dataframe['startTime' if key != 'kpi' else 'timestamp'] >= timestamp]
            
            print(df['esb'])
            # Detect anomalies on esb
            esb_anomaly_res = []
            try:
                esb_anomaly_res = detect_esb.find_anom(df['esb'])
            except: 
                pass
            
            print(esb_anomaly_res)

        else:  # message.topic == 'trace'
            new_df = Trace(data).to_dataframe() 
            df['trace'] = pd.concat([df['trace'],new_df], ignore_index=True)



if __name__ == '__main__':
    main()