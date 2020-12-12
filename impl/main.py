'''
Example for data consuming.
'''
import json
import os

import pandas as pd
import requests
from kafka import KafkaConsumer

import threading

# from models.esb_detection.vae import detect as esb_vae
from models.esb_detection.seas_decomp import detect as esb_seas
from models.trace_localisation.microrca.micro_rca import MicroRCA
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


def handle_anomaly(df, microRCA):
    # Local testing only. Load some sample data
    if SERVER_CONFIGURATION["SUBMIT_IP"] is None:
        micro_rca_data_dir = os.path.join('models','trace_localisation','microrca','data')
        if df['trace'].empty:
            df['trace'] =  pd.read_csv(os.path.join(micro_rca_data_dir, 'small_trace.csv')).drop(['Unnamed: 0'], axis=1)
        if df['kpi'].empty:
            df['kpi'] = pd.read_csv(os.path.join(micro_rca_data_dir, 'small_kpis.csv')).drop(['Unnamed: 0'], axis=1)

    anomalous_hosts = microRCA.detect(df['trace'], df['kpi'])
    
    print(anomalous_hosts)

    # >> Complex piece of logic here... 
    # def host_to_key(host):
    #     if 'docker_' in host: return 'dcos_docker'
    #     if 'container_' in host: return 'dcos_container'
    #     if 'db_' in host: return 'db_oracle_11g'
    #     if 'redis_' in host: return 'mw_redis'
    #     if 'os_' in host: return 'os_linux'
    #     return host # 

    # kpi_dict = dict(tuple(df['kpi'].groupby('cmdb_id')))
    # for key in list(kpi_dict.keys()):
    #     new_key = host_to_key(key)
    #     kpi_dict[new_key] = kpi_dict.pop(key)

    # print(kpi_dict)
    # print(df['kpi'])
    # # Run KPI anomaly detection for each host in order
    # host_kpi_anomalies = []
    # for host in anomalous_hosts:
    #     print('Running KPI detection on host', host)

    #     problems = kpi_detection.find_anom(host, {host_to_key(host): df['kpi']})
    #     host_kpi_anomalies.append(problems)
    
    # print(host_kpi_anomalies)

    # if no host is anomalus, do submit the first from the list with None
    # NOTE: Do we just select one? The submit function also accepts a list of [(host, kpi), (host, kpi),...]
    if SERVER_CONFIGURATION["SUBMIT_IP"] is not None:
        
        submit(anomalous_host) 


def main():
    '''Consume data and react'''
    # Check authorities
    assert AVAILABLE_TOPICS <= CONSUMER.topics(), 'Please contact admin'

    df = {
        'esb': pd.DataFrame(columns=['serviceName','startTime','avg_time','num','succee_num','succee_rate']), 
        'trace': pd.DataFrame(columns=['startTime','elapsedTime','success','traceId','id','pid'',cmdb_id','serviceName','callType']), 
        'kpi': pd.DataFrame(columns=['item_id','name','bomc_id','timestamp','value','cmdb_id'])
    }
    # Initialize the MicroRCA detector
    microRCA = MicroRCA()
    i = 0
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
            print(f'Received {data}')
            new_df = pd.concat(map(lambda x: x.to_dataframe(), [BusinessIndex(item) for key in data['body'] for item in data['body'][key]]))
            df['esb'] = pd.concat([df['esb'],new_df], ignore_index=True)

            window_size_ms = 1000 * 60 * 5 # window size: 5min
            timestamp = int(new_df.iloc[-1]['startTime']) - window_size_ms # check the most recent
    
            # remove from all dfs
            for key in df:
                dataframe = df[key]
                df[key] = dataframe[dataframe['startTime' if key != 'kpi' else 'timestamp'] >= timestamp]
          

            # Detect anomalies on esb with seasonality decomposition
            esb_is_anomalous = esb_seas.find_anom(df['esb'])

            if(esb_is_anomalous):
                print('Anomaly detected. Tracing...')
                t = threading.Thread(target=handle_anomaly, args=(df,microRCA))
                t.start()
                # TODO do we need to wait for it?

        else:  # message.topic == 'trace'
            new_df = Trace(data).to_dataframe() 
            df['trace'] = pd.concat([df['trace'],new_df], ignore_index=True)



if __name__ == '__main__':
    main()
