# Configuring

Please make sure to create a file called `server_config.py` which must be a duplicate of the given example.
The file contains the following structure:

```
SERVER_CONFIGURATION = {
    "KAFKA_QUEUE" : <QUEUE IP>,
    "SUBMIT_IP" : <SUBMISSION IP>
}
```

The idea is to not require individual tuning on the server and locally once done.


## Install Kafka

Please make sure to run `python3 -m pip install kafka-python` so that you are able to run Kafka.

Then, you'll need to create a local Kafka Server to run the server, this includes a Zookeeper. You can check the installation guide at [the official website](https://kafka.apache.org/quickstart) to install.

1. Download the latest kafka release
2. Unpack kafka in a directory and `cd` to inside it
3. $ ./bin/zookeeper-server-start.sh ./config/zookeeper.properties
4. $ ./bin/kafka-server-start.sh ./config/server.properties


## Running 

1. Run the `server.py` in this repository: `python3 server.py`
2. Run the `main.py` file. You should see some training done by the VAE and then it starting. The server is sending a message at every 5s.