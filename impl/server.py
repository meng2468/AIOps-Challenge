from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic

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

    json_object = json.dumps(
    {   "startTime":16666587823,
        "body": { 
            "esb": [{
                "serviceName": "test",
                "startTime": 16666587823,
                "avg_time": 1.53,
                "num": 350,
                "succee_num": 175,
                "succee_rate": 175 / 350
            }]
        }
    }).encode('utf-8')
    while True:
        producer.send('business-index', value=json_object)
        print('ZZZZzzzz')
        sleep(5)