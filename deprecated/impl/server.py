from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic

import time
from time import sleep
import json
import pandas as pd
import os

if __name__ == '__main__':
    DATA_PATH = os.path.join('..','data','test_data')

    # esb_df = pd.read_csv(os.path.join(DATA_PATH, 'esb.csv'))
    
    try:
        admin_client = KafkaAdminClient(
            bootstrap_servers="localhost:9092", 
            client_id='test'
        )
        AVAILABLE_TOPICS = set(['platform-index', 'business-index', 'trace'])

        topic_list = [NewTopic(name=name, num_partitions=1, replication_factor=1) for name in AVAILABLE_TOPICS]
        admin_client.create_topics(new_topics=topic_list, validate_only=False)
    except:
        print('Topics already created, proceed')
    
    producer = KafkaProducer(bootstrap_servers='localhost:9092')

    # def get_esb_json(esb_row):
    #     # delta_time = int(time.time() - start) * 1000

    #     json_object = json.dumps(
    #     {   "startTime": esb_row['start_time'],
    #         "body": { 
    #             "esb": [{
    #                 "serviceName": esb_row['service_name'],
    #                 "startTime": esb_row['start_time'],
    #                 "avg_time": esb_row['avg_time'],
    #                 "num": esb_row['num'],
    #                 "succee_num": esb_row['succee_num'],
    #                 "succee_rate": esb_row['succee_rate']
    #             }]
    #         }
    #     }).encode('utf-8')
    #     return json_object

    send_interval = 5

    def get_json(start):
        delta_time = int(time.time() - start) * 1000 * (60/send_interval) # Force increments to be 1 min
        json_object = json.dumps(
        {   "startTime":1606862220032 + delta_time,
            "body": { 
                "esb": [{
                    "serviceName": "test",
                    "startTime": 1606862220032 + delta_time,
                    "avg_time": 1.53,
                    "num": 350,
                    "succee_num": 175,
                    "succee_rate": 175 / 350
                }]
            }
        }).encode('utf-8')
        return json_object

    start = time.time()

    while True:
        producer.send('business-index', value=get_json(start))
    # for index, row in esb_df.iterrows():
    #     row_json = get_esb_json(row)
    #     producer.send('business-index', value=row_json)
        print('ZZZZzzzz')
        # print('[INFO] Sent ESB:', row_json)
        sleep(send_interval)