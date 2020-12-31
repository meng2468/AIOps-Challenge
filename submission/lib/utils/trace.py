from .data_types import Trace, BusinessIndex, PlatformIndex
import numpy as np

from collections import defaultdict

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
        csf_items = {} # {<id>: <object reference>}

        for i in range(len(trace) - 1, 0, -1):
            times[trace[i].pid] += trace[i].elapsed_time
            if trace[i].call_type == 'CSF':
                csf_items[trace[i].id] = trace[i]
        
        for el in trace:
            el.elapsed_time -= times[el.id]
            if el.pid in csf_items:
                csf_items[el.pid].service_name = el.cmdb_id

    return trace_dict

def get_anomalous_hosts_count(limits, traces):
    traces = parse(traces)
    results = defaultdict(lambda: [0,0])

    anomalous_trace_count = 0
    for trace_id, trace in traces.items(): 
        is_anomalous = False
        for trace_span in trace:
            # Check if threshold is surpassed by the elements
            key = (trace_span.service_name)
            if not limits[key]:
                # print(key,'not found')
                continue
            
            lower, upper = limits[key]
            if not lower <= trace_span.elapsed_time <= upper or trace_span.success == False:
                results[key][0] += 1
                is_anomalous = True
            results[key][1] += 1
            if isinstance(trace_span.success, str):
                raise ValueError('Sccess is string')

        if is_anomalous:
            anomalous_trace_count += 1

    return anomalous_trace_count / len(traces), results # percentage of traces, individual counts