# Implementation details

- Dockerfile is the file to generate the docker container
    - To build the container do `docker build -t aiops .`
- main.py is the file that communicates with Kafka
    - Kafka is a stream where events will be handed out during 12h. There are 3 categories (platform-index, business-index, trace) from where it reads
    - There are 3 classes, one for each category. Each class contains the attributes of the given type (described in the problem statement)


### TODO

- Understand more about kafka
- Create a kafka broker for testing (optional)
