from .data_types import Trace, BusinessIndex, PlatformIndex
import numpy as np

from collections import defaultdict

import pandas as pd
import time
import sys
import random

def parse(traces):
    """
    Input:
        traces - list of Trace objects as in main.py

    Output:
        dictionary {<trace_id> : [<processed traces>]}

    Processing includes sorting and removing the time of the children from the elapsed time of the parent
    """
    
    trace_dict = defaultdict(list) # { trace_id : [traces]}

    # Split traces according to their id
    for trace in traces:
        trace_dict[trace.trace_id].append(trace)

    # Sort them according to start time
    for trace in trace_dict.values():
        trace.sort(key=lambda x: x.start_time)

    # Remove time of the children from the parent
    for trace in trace_dict.values():
        times = defaultdict(int)
        
        for i in range(len(trace) - 1, 0, -1):
            times[trace[i].pid] += trace[i].elapsed_time
        
        for el in trace:
            el.elapsed_time -= times[el.id]

    return trace_dict


if __name__ == '__main__':
    print(f'Testing {__file__}')

    if len(sys.argv) >= 2:
        nrows = sys.argv[1]
    else:
        nrows = 300000
    print(f'Load test will have {nrows} spans')

    if len(sys.argv) >= 3:
        seed = sys.argv[2]
    else:
        seed = 69420 #lmao

    trace = pd.read_csv('../server_data/trace.csv', nrows=10000).reset_index()
    
    traceid = trace.at[1500, 'traceId']
    
    single_trace = trace[trace.traceId == traceid]
    res = []
    single_trace.apply(lambda x: res.append(Trace(x)), axis=1)
    random.shuffle(res)
    assert len(res) == len(single_trace), 'Error processing trace: transformation of different size'

    res = parse(res)
    trace = res[traceid]

    start_time = 0
    for trace_el in trace:
        assert trace_el.start_time >= start_time
        start_time = trace_el.start_time

    print('Integrity check passed.')
    
    trace = pd.read_csv('../server_data/trace.csv', nrows=nrows).reset_index()
    traces = []
    trace.apply(lambda x: traces.append(Trace(x)), axis=1)
    random.shuffle(traces)
    print('Initiating load test')

    start = time.time()
    parse(traces)
    end = time.time()
    
    print(f'Took {end - start}s to process {nrows} spans')
