from itertools import groupby

def trace_to_call_path(df):
    """
    df : Trace Pandas dataframe. The dataframe must be sorted and pre processed
    Pre processing includes dsName being removed from JDBC and LOCAL
    """
    
    # transform CSF serviceName
    ids = df[df['callType'] == 'CSF']['id']
    children_cmdb = df[df['pid'].isin(ids)]['cmdb_id']
    df.loc[df['callType']=='CSF', 'serviceName'] = list(children_cmdb)


    names = dict(zip(df.id, df.serviceName))
    names['None'] = 'Start'

    path = []

    def merge_fn(row):
        """ Makes a row become (pid, id) format while filtering same service calls"""
        v = (names[row['pid']], names[row['id']])
        if v[0] == v[1]:
            return
        path.append(v)
    
    df.apply(merge_fn, axis=1) # apply horizontally
    l = path
    path = [x[0] for x in groupby(path)]
    times = df['elapsedTime'].values.tolist()
    call_path = set()
    for index, p in enumerate(path):
        call_path.add((p[1], tuple(path[:index + 1]), times[index]))

    return call_path
