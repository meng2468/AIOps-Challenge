'''
Example for data consuming.
'''
import requests
import json
import datetime
import csv
from kafka import KafkaConsumer

# Three topics are available: platform-index, business-index, trace.
# Subscribe at least one of them.
AVAILABLE_TOPICS = set(['platform-index', 'business-index', 'trace'])
CONSUMER = KafkaConsumer('platform-index', 'business-index', 'trace',
                         bootstrap_servers=['172.21.0.8', ],
                         auto_offset_reset='latest',
                         enable_auto_commit=False,
                         security_protocol='PLAINTEXT')

def main():
    '''Consume data and react'''
    # Check authorities
    assert AVAILABLE_TOPICS <= CONSUMER.topics(), 'Please contact admin'
    i = 0
    for message in CONSUMER:
        i += 1
        folder = 'logs'
        date_pref = datetime.datetime.now().strftime('%d-%m:%H')
            
        data = json.loads(message.value.decode('utf8'))
        if message.topic == 'platform-index':
            with open(date_pref+'_host.csv', 'a') as csv_file:
                for stack in data['body']:
                    for item in data['body'][stack]:
                        csv.writer(csv_file).writerow([str(x) for x in item.values()])
            timestamp = data['timestamp']
        elif message.topic == 'business-index':
            with open(date_pref+'_esb.csv', 'a') as csv_file:
                for stack in data['body']:
                    for item in data['body'][stack]:
                        csv.writer(csv_file).writerow([str(x) for x in item.values()])            
            timestamp = data['startTime']
        else:
            with open(date_pref+'_trace.csv', 'a') as csv_file:
                csv.writer(csv_file).writerow([str(x) for x in data.values()])    
            timestamp = data['startTime']
        print(i, message.topic, timestamp)


if __name__ == '__main__':
    main()