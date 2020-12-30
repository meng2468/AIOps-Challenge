import pandas as pd
import time
import sys
import random

from utils.data_types import *
from utils.trace import *

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