### Submission

This is the directory which will run on the server
The dataset is available in [Mega](https://mega.nz/file/scoDXQIK#49cYHKAPexpakrI0lJAhS31rOvNz4nKKUmEzEZMCanA)

```
├── lib                 # Contains the utility files to the main program
│   ├── models          # Contains the quantiles file each one with different types and quantiles
│   └── utils           # Utility files including 
└── server_data         # Where the test data of the server should be
```
## Running 

1. Run `python3 server.py [test files directory]` to start serving test data. If no argument is provided, it will default to `server_data/`. You need to have a Kafka server running (see below).
2. Run `python3 main.py <quantile file>`. You should see some output information everytime there's an ESB message. The server is sending the messages as fast as possible. Quantiles are available in `lib/models/`.

The configurations are as below.

# Configuring

Please make sure to create a file called `server_config.py` which must be a duplicate of the given example `server_config.py.example`.
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
3. `$ ./bin/zookeeper-server-start.sh ./config/zookeeper.properties`
4. `$ ./bin/kafka-server-start.sh ./config/server.properties`