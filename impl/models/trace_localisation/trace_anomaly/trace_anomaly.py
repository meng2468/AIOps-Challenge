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
            data = list(map(lambda x: (x[0],eval(x[1])), list(reader)))        
            
        # path : position
        self.paths = {data[i]: i for i in range(len(data))}


    def get_stv(self, dataframe):
        """dataframe should be a single trace already pre processed"""
        graph = list(utils.trace_to_call_path(dataframe))
        
        # set invalid dimensions and add the valid ones
        res = [-1 for _ in range(len(self.paths))]

        # (s, path, time)
        indexes = tuple(map(lambda x: self.paths[x[:2]], graph))
        print(indexes)

        # NOTE there is a possibility for a sparse representation in case of lack of memory
        # id : val
        for i in range(len(graph)):
            res[indexes[i]] = graph[i][2]
        
        return res

