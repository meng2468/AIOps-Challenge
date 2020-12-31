### Generates and Saves VAE Models and Anomaly Threshold for each KPI

import numpy as np
import pandas as pd

import preprocessing as pp
import model_gen as vae

time_step = 432

data_path = 'train_data/host/'
save_dir = 'deploy/models_' + str(time_step) + '/'
thresh_dir = 'deploy/'

dfs = pp.load_dfs(data_path)

models = []
histories = []

def get_threshold(model, x_train):
    x_train_pred = model.predict(x_train)
    train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)
    threshold = np.max(train_mae_loss)
    return threshold

df_thresh = pd.DataFrame(columns=['KPI', 'thresh'])

for key in dfs:
    df = dfs[key]
    for name in list(df['name'].unique()):
        #Train and save the models
        df_n = df[df.name == name]
        x_train  = pp.get_kpi_train_data(df_n, time_step)
        if x_train == []:
            print(name, ' has no data! Skipping model generation!')
            continue
        model, history = vae.save_model(x_train, save_dir, key, name)
        models.append(model)
        histories.append(history)

        thresh = {'KPI': key+'_'+name, 'thresh': get_threshold(model, x_train)}
        df_thresh = df_thresh.append(thresh, ignore_index=True)
    df_thresh.to_csv(thresh_dir+'thresh_'+str(time_step)+'.csv', index=False)