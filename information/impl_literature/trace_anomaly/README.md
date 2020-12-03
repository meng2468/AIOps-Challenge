# Paper Notes
Thoughts / Summaries on Relevant Papers

#### Root Cause Detection in a Service-Oriented Architecture
URL: https://netman.aiops.org/~peidan/ANM2020/7.TraceAnomalyDetection/LectureCoverage/2013SIGMETRICS13_Root%20Cause%20Detection%20in%20a%20Service-Oriented%20Architecture.pdf

##### MonitorRank
- Provides ranking of possible root causes
- Unsupervised model uses
    - Historical time-series metrics
    - Current time-series metrics
    - Call graph 

##### Call Graph Issues 
- Might not represent true dependencies
- Does not account for external factors
- How do we handle serial and parallel calls?
>For example, call latency for a service would be greater than the sum of its downstream calling service’s latency for serial calls, but be maximum in case of parallel calls.
- Dependencies change over time

##### System Subcomponents
MonitorRank consists of three different components
###### Metrics Collection 
- Consumes Kafka data
- Buffers and aggregates to coarser time granularity (how coarse?)
- Stores into time-partitioned database

###### Batch-Mode Engine 
- Generates call graph with a HADOOP job
- Clusters sensors together based on historical correlation

###### Real-time Engine 


#### Latent Error Prediction and Fault Localization for Microservice Applications by Learning from System Trace Logs
URL: https://github.com/meng2468/anm-project/blob/main/information/impl_literature/trace_anomaly/Latent%20Error%20Prediction%20and%20Fault%20Localization%20for%20Microservice%20Applications%20by%20Learning%20from%20System%20Trace%20Logs.pdf

###### Approach (RIP)
>To train the models, we need not only trace logs under successful executions but also trace logs under erroneous executions. 
