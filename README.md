# AIOps Challenge 2020

This project was done under the scope of the Advanced Network Management course at Tsinghua University.

Project report is available in `report.pdf` and contains our analysis of the problem, as well as an explanation of what's happening in the submission under `submission/`.

## Code organization
The repository is organized in the following 3 folders:
- data: Contains the competition dataset.
- deprecated: Contains all the failed approaches before reaching the final solution.
- submission: Contains the code of the final solution.


```
├── data                         
│   ├── full_data               # Functions to process and parse data 
├── deprecated                  # Old solutions
│   ├── data_processing         # Notebooks to process trace information        
│   ├── eda                     # Notebooks for data analysis and visualization
│   ├── impl                    # Deployment
├── information                 # Documentation
└── submission                  # Final solution in root folder
```

## Quick start
To run this project, we recommend you to have Python >= 3.6 installed in your system.
Navigate to the submission folder to check the final submission.

To run the main program, simply execute this command:
```
python3 main.py
```

(Optional) Start Kafka and Zookeeper locally to perform the local testing. After both are running, execute the server to stream the test data:

```
python3 server.py
```

## Project Resources
- https://github.com/NetManAIOps/aiops2020-judge/tree/master/final (Competition Github)
- https://cloud.tsinghua.edu.cn/f/e06aaab7135c44e8beec/?dl=1 (Conmpetition Slides in Chinese)
- http://81.70.98.179:8000/standings/show/ (Ranking)

## Structure of the challenge
We are given 3 types of data (ESB, Trace and KPI) and we are supposed to implement a program that is able to timely detect the anomalies.
The anomaly consists of a collection of pairs (host, kpi).

Constraints:
- Data is served in a Kafka Queue
- No GPU support