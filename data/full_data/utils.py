import pandas as pd
import numpy as np

def get_esb(anomaly):
    esb = pd.read_csv(anomaly['folder']+'/esb.csv')
    esb.head()