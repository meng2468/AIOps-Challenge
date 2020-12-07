# ANM Project
## Project Resources
- Elearning documents in /Information
- https://github.com/NetManAIOps/aiops2020-judge/tree/master/final (Competition Github)
- https://cloud.tsinghua.edu.cn/f/e06aaab7135c44e8beec/?dl=1 (Conmpetition Slides)
- http://81.70.98.179:8000/standings/show/ (Ranking)

## Task
### Anomaly Detection
Compute a real-valued score indicating the certainty of having an anomaly, given historical data. 

Human operators can then affect whether to declare an anomaly by choosing a threshold, where a data point with a score exceeding this threshold indicates an anomaly.

### TroubleShooting
Find out system nodes (vm or docker) and KPIs where the root cause occurs when a failure happens.

The troubleshooting in the figure has 3 steps:
1. Find the time point when the business success rate was significantly lower than 1. (ESB)
2. Around the time point, look into the anomalous behaviors of microservices and record containers or hosts where the microservices are deployed. (Trace)
3. After the abnormal nodes, hosts or containers, are found.  Detect which KPIs of the nodes perform anomalously. (KPI)

## Data 
### ESB business indicator(ESB)
Minute-wise request information for osb_001
- startTime: aggr. info on all requests for the following minute
- avg_time: average time spent processing a request.
- num: number of requests
- succee_num: the number of submitted requests which are successfully completed.
- succee_rate: succee_num / num

EDA: https://github.com/meng2468/anm-project/blob/main/eda/esb.ipynb

### Trace
- id: span id
- pid: parent span id
- tracid: id of trace the span belongs to
- cmdb_id: host
- callType: csv's are split based on this
    - osb, remoteprocess, flyremote inside span
    - csf, local, jdbc outside of span
- success: whether service is processed succesfully
- dsName: database accesed by microservice

EDA: https://github.com/meng2468/anm-project/blob/main/eda/trace.ipynb

### Host KPIs data
Time series KPI (Key Performance Indicator) data for different hosts
- itemid: KPI identifier?
- name: KPI identifier?
- bomc_id: KPI identifier?
- timestamp: measurement time
- value: KPI value
- cmdb_id: host name
