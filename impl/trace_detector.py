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
        # print(group)
        series = groupdf['elapsedTime'].resample('60S').mean()
        val = esd.esd_test(series, 6) # frequency should be different?
        # print(val[1])
        
        host, service = group
        table.at[service,host] = val[1]

    table = table.fillna(0)
    print(table)

    threshold = table.values.max() / 10
    print(threshold)
    values = np.where(table.values > threshold)
    
    pairs = []
    pairs.append(list(map(lambda x: table.index[x], values[0])))
    pairs.append(list(map(lambda x: table.columns[x], values[1])))
    
    pairs = [x for x in zip(*pairs)]
    print(pairs)
    


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
    detect(traces,kpis)