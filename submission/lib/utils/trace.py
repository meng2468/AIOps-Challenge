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