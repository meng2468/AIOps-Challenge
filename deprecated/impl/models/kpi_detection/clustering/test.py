import pandas as pd
import os
import datetime

import preprocessing_rocka
import baseline_extraction
import shape_based_distance
import density_estimation
import model

if __name__ == '__main__':
    PATH = '../../../../data/training_data/host/'
    files = [PATH + f for f in os.listdir(PATH)]
    df = pd.concat([pd.read_csv(f) for f in files])

    kpis = dict(tuple(df.groupby(['cmdb_id','name'])))
    
    key = ('docker_008', 'container_cpu_used')
    kpidf = kpis[key]
    kpidf['timestamp'] = kpidf['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000.0))
    kpidf = kpidf.set_index('timestamp')['value']
    kpidf = kpidf.resample('T').mean()
    kpidf = kpidf.interpolate()
    d = kpidf.values
    d = (d - d.mean()) / d.std()
    
    
    d = baseline_extraction.smoothing_extreme_values(d)
    d = baseline_extraction.extract_baseline(d, 5) # 5 is the time window size, i set it to 5 minutes...


    # Previously is the standard way to prepare a KPI
    # After needs to get the distance between multiple KPIs
    # And then find the density
    # After that, we can model the DBSCAN