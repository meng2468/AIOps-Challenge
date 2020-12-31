import pandas as pd
import numpy as np

import os
import datetime

import preprocessing_rocka
import baseline_extraction
import shape_based_distance
import density_estimation
import model

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
d = baseline_extraction.extract_baseline(d, 5) 
