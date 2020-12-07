import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

from . import utils

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

    @property
    def n_dim(self):
        return len(self.paths)

    def to_dense_stv(self, sparse_stv):
        """
        Convert a dictionary STV (sparse) into a dense vector of dimension n_dim = len(self.paths)
        """
        dense_stv = np.zeros(self.n_dim)
        for key, value in sparse_stv.items():
            dense_stv[key] = value
        return dense_stv

    def get_stv(self, dataframe, sparse=True):
        """dataframe should be a single trace already pre processed"""
        graph = list(utils.trace_to_call_path(dataframe))
        
        # (s, path, time)
        indexes = tuple(map(lambda x: self.paths[x[:2]], graph))

        # id : val
        sparse_stv = dict(zip(indexes, tuple(map(lambda x: x[2], graph))))
        return sparse_stv if sparse else self.to_dense_stv(sparse_stv)