import pandas as pd
import datetime
from collections import defaultdict
import numpy as np
import ts_anomaly_detection as esd

def detect(traces,kpis):
    traces = dict(tuple(traces.groupby('traceId')))
    print(f'Number of traces {len(traces)}')
    for i, trace in enumerate(traces):
        if i+1 % 500 == 0:
            print(i)
        traces[trace] = process(traces[trace]).drop('dsName', axis=1)
        # print(traces[trace])
    
    df = pd.concat(traces.values())
    print(df)
    groups = df.groupby(['cmdb_id','serviceName'])
    table = pd.DataFrame(columns=sorted(list(df['cmdb_id'].unique())), index=sorted(list(df['serviceName'].unique())))
    # print(table)
    for group, groupdf in groups:
        print(group)
        series = groupdf['elapsedTime'].resample('60S').mean()
        val = esd.esd_test(series, 6) # frequency should be different?
        print(val[1])
        
        host, service = group
        table.at[service,host] = val[1]

    print(table)

def process_kpis(kpis_df):
    kpis_df['timestamp'] = kpis_df['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x/ 1000.0))
    kpis_df = kpis_df.set_index('timestamp')
    return kpis_df
    
def get_kpi_given_host(host, kpis):
    # Get only the kpis of that host
    kpis = kpis[kpis['cmdb_id'] == host]
    kpis = process_kpis(kpis)
    
    groups = kpis.groupby(['name'])
    table = pd.DataFrame(columns=[host], index=sorted(list(kpis['name'].unique())))

    for group, groupdf in groups:
        series = groupdf['value'].resample('60S').mean()
        series = series.fillna(0)
        # print(group, series)
        val = esd.esd_test(series, 6) 
        kip_name = group
        table.at[group, host] = val[1]
    
    print(table)

def process(traces_df):
        ids = traces_df[traces_df['callType'] == 'CSF']['id'].values

        traces_df['startTime'] = traces_df['startTime'].apply(lambda x: datetime.datetime.fromtimestamp(x/ 1000.0))
        traces_df = traces_df.set_index('startTime')
        # print(traces_df)
        # print(ids)
        children_times = defaultdict(list)
        

        relationship = {}
        
        def parse(row):
            # parent -> child
            if row['pid'] in ids:
                relationship[row['pid']] = row['cmdb_id']

            children_times[row['pid']].append(row['elapsedTime'])

            if row['callType'] in ['LOCAL','JDBC']:
                row['serviceName'] = row['dsName']
            
            elif row['callType'] == 'OSB' or row['callType'] == 'RemoteProcess':
                row['serviceName'] = row['cmdb_id']

            return row

        def apply(row):
            # time of current becomes time of current minus children
            
            row['elapsedTime'] = row['elapsedTime'] - sum(children_times[row['id']])

            # parent -> new_parent
            if row['callType'] != 'CSF':
                return row
            else:
                if row['id'] in relationship:
                    row['serviceName'] = relationship[row['id']]
                return row

        traces_df = traces_df.apply(parse, axis=1) # transform dsName
        return traces_df.apply(apply, axis=1)


if __name__ == '__main__':
    import sys
    path = sys.argv[1]
    kpis = pd.read_csv(path + '/kpi.csv')
    traces = pd.read_csv(path + '/trace.csv')
    # detect(traces,kpis)
    get_kpi_given_host('docker_005', kpis)