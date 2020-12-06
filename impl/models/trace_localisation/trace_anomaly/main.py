import pickle
from pprint import pprint
import sys

import utils
import trace_anomaly

EDA_DIR = '../../../../eda/'
DATA_DIR = '../../../../data/'

if __name__ == '__main__':
    # with open (EDA_DIR + 'traces.pickle', 'rb') as f:
    #     traces = pickle.load(f) # load traces
    
    """
    TODO
    1 - TODO need to process traces to build each trace's STV
    2 - TODO they need to be put in a searchable format (ont-hot?)
        

    1- Build trace into format
    2- Check if it fits any trace
        2.1- if no, anomaly
    3- Run anomaly detection in all traces from that interval
    4- For the ones with less score, localize root cause
    """

    with open(sys.argv[1], 'rb') as f:
        trace = pickle.load(f)
    
    #trace = list(utils.trace_to_call_path(trace))

    model = trace_anomaly.TraceAnomaly(DATA_DIR + 'trace_list.csv')
    print(model.get_stv(trace))
