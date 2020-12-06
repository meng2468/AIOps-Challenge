import torch
import torch.nn as nn
import torch.functional as F

import utils

class TraceAnomalyDetector(nn.Module):

    def __init__(self):
        pass

    def forward(self):
        pass


class TraceAnomaly:
    def __init__(self, paths_csv_file):
        import csv
        with open(paths_csv_file) as f:
            reader = csv.reader(f)
            # transform second position into a tuple instead of string
            data = sorted(list(map(lambda x: (x[0],eval(x[1])), list(reader))))
            
        # path : position
        self.paths = {data[i]: i for i in range(len(data))}


    def get_stv(self, dataframe):
        """dataframe should be a single trace already pre processed"""
        graph = list(utils.trace_to_call_path(dataframe))
        
        # (s, path, time)
        indexes = tuple(map(lambda x: self.paths[x[:2]], graph))

        # id : val
        return dict(zip(indexes, tuple(map(lambda x: x[2], graph))))

