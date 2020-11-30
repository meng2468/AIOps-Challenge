# -*- coding: utf-8 -*-
"""Example of using Variational Auto Encoder for outlier detection
"""
# Author: Andrij Vasylenko <andrij@liverpool.ac.uk>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys

from pyod.models.vae import VAE
from pyod.utils.data import evaluate_print
from joblib import dump, load

import pandas as pd

if __name__ == "__main__":
    esb_data = pd.read_csv('../data/training_data/2020_05_04/esb.csv')
    print(esb_data)
    esb_data = esb_data[['avg_time', 'succee_rate']]
    print(esb_data)

    esb_data = esb_data.append(pd.DataFrame({'avg_time': [1.4], 'succee_rate': [0.5]})) # force bad case for improved performance

    # train VAE detector (Beta-VAE)
    clf_name = 'VAE'
    clf = VAE(encoder_neurons=[2,1], decoder_neurons=[1,2], epochs=30, contamination=1/len(esb_data), gamma=0.8, capacity=0.2)
    clf.fit(esb_data)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    # get the prediction on the test data
    y_test_pred = clf.predict(esb_data)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(esb_data)  # outlier scores

    evaluate_print(clf_name, [0 for _ in range(len(y_test_scores)-1)] + [1], y_test_scores)

