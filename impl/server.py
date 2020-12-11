from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic

import time
from time import sleep
import json

if __name__ == '__main__':
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

    def get_json(start):
        delta_time = int(time.time() - start) * 1000
        json_object = json.dumps(
        {   "startTime":16666587823 + delta_time,
            "body": { 
                "esb": [{
                    "serviceName": "test",
                    "startTime": 1606881660000 + delta_time,
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
        print('ZZZZzzzz')
        sleep(1)