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
    # for trace_id, trace in traces.items(): 
    #     is_anomalous = False
    #     for trace_span in trace:
    #         # Check if threshold is surpassed by the elements
    #         key = (trace_span.service_name)
    #         if not limits[key]:
    #             # print(key,'not found')
    #             continue
            
    #         lower, upper = limits[key]
            # if not lower <= trace_span.elapsed_time <= upper or trace_span.success == False:
            #     results[key][0] += 1
            #     is_anomalous = True
            # results[key][1] += 1
            # if isinstance(trace_span.success, str):
            #     raise ValueError('Sccess is string')

        # if is_anomalous:
        #     anomalous_trace_count += 1


    missing_keys = []
    for trace_id, trace in traces.items():
        
        # generate trace tree depth
        depth = {'None' : 0}
        trace.sort(key=lambda x: x.start_time)
        for trace_span in trace:
            if trace_span.pid not in depth:
                continue

            trace_span.depth = depth[trace_span.pid] + 1
            depth[trace_span.id] = trace_span.depth

        for trace_span in trace:
            if trace_span.pid not in depth: # data missing atm
                trace_span.depth = -1
                continue
            trace_span.depth = depth[trace_span.pid] + 1
            depth[trace_span.id] = trace_span.depth

        if len(list(limits.keys())[0]) == 2:
            # no host in the key
            get_key = lambda x: (x.depth, x.call_type)
        else:
            get_key = lambda x: (x.depth, x.call_type, x.cmdb_id)
        

        is_anomalous = False
        # detect individual anomalies
        for trace_span in trace:
            key = get_key(trace_span)
            if key not in limits:
                # print(key)
                continue
            
            lower, upper = limits[key]
            if not lower <= trace_span.elapsed_time <= upper or trace_span.success == False:
                results[trace_span.service_name][0] += 1
                is_anomalous = True
            results[trace_span.service_name][1] += 1
        
        if is_anomalous:
            anomalous_trace_count += 1

    return anomalous_trace_count / len(traces), results # percentage of traces, individual counts
