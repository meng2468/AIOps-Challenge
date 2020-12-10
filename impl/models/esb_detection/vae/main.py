### Generates and Saves VAE Models and Anomaly Threshold for each KPI

import numpy as np
import pandas as pd

import preprocessing as pp
import model_gen as vae

time_step = 24

data_path = 'train_data/'
save_dir = 'deploy/'
thresh_dir = 'deploy/'

df = pp.load_esb(data_path)
def get_threshold(model, x_train):
    x_train_pred = model.predict(x_train)
    train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)
    threshold = np.max(train_mae_loss)
    return threshold

df_thresh = pd.DataFrame(columns=['name', 'thresh'])

for name in ['num', 'avg_time']:
    #Train and save the models
    x_train  = pp.get_esb_train_data(df, name, time_step)
    if x_train == []:
        print('Not enough data for timestep!')
        continue
    model, history = vae.save_model(x_train, save_dir, name)

    thresh = {'name': name, 'thresh': get_threshold(model, x_train)}
    df_thresh = df_thresh.append(thresh, ignore_index=True)
df_thresh.to_csv(thresh_dir+'thresh.csv', index=False)