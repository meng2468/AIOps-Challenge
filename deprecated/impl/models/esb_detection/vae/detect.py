import numpy as np
import pandas as pd
import os

from . import preprocessing as pp
from . import model_gen as vae

from tensorflow import keras
from tensorflow.keras import layers

time_step = 24

data_path = 'train_data/'
model_dir = 'deploy'
thresh_dir = 'deploy'


def find_anom(df):
    problems = []

    # Make sure the directories work when invoked from the main script
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    thresh_filename = os.path.join(dir_path, thresh_dir, 'thresh.csv')
    thresh = pd.read_csv(thresh_filename)

    for name in ['num', 'avg_time']:
        x_test = pp.get_esb_train_data(df, name, time_step)

        model_filename = os.path.join(dir_path, model_dir, name)
        model = keras.models.load_model(model_filename)

        x_test_pred = model.predict(x_test)
        test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
        test_mae_loss = test_mae_loss.reshape((-1))
        anomalies = np.greater(test_mae_loss, thresh[thresh.name==name]['thresh'].values[0])

        if True in anomalies:
            print("Anomaly in ", name)
            problems.append(name)

        if problems == []:
            print("No anomalies found in ", name)
    return problems

if __name__ == "__main__":
    find_anom(pp.load_esb(data_path))