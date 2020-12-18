import pandas as pd
import datetime
from collections import defaultdict

import ts_anomaly_detection as esd

def detect(traces,kpis):
    traces = dict(tuple(traces.groupby('traceId'))[:50])
    print(f'Number of traces {len(traces)}')
    for i, trace in enumerate(traces):
        if i+1 % 500 == 0:
            print(i)
        traces[trace] = process(traces[trace]).drop('dsName', axis=1)
        # print(traces[trace])
    
    df = pd.concat(traces.values())

    groups = df.groupby(['cmdb_id','serviceName'])
    for group, groupdf in groups:
        # print(groupdf.index.dtype)
        series = groupdf['elapsedTime'].resample('10S').mean().fillna(0)
        val = esd.esd_test(series, 6) # frequency should be different?
        print(val, type(val[1]))

    

    


def process(traces_df):
        ids = traces_df[traces_df['callType'] == 'CSF']['id']

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
            
            elif row['callType'] == 'OSB':
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
                    row['cmdb_id'] = relationship[row['id']]
                return row

        traces_df = traces_df.apply(parse, axis=1) # transform dsName
        return traces_df.apply(apply, axis=1)


if __name__ == '__main__':
    kpis = pd.read_csv('test.csv')
    traces = pd.read_csv('trace_test.csv')
    detect(traces,kpis)