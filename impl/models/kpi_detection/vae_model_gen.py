import numpy as np
import pandas as pd
import os

from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

data_path = '../../../data/train_data/host'
save_dir = '' # Model save-file directory
dfs = {}
   
for file in os.listdir(data_path):
    print('Saving ' + file[:-4] + ' into dfs')
    dfs[file[:-4]] = pd.read_csv(data_path+'/'+file) 

def normalise(df):
    mean = df['value'].mean()
    std = df['value'].std()
    if std == 0:
        df['value'] = df['value'] - mean
        return df
    
    df['value'] = (df['value'] - mean) / std
    return df

def gen_train_seq(values, time_steps=288):
    output = []
    for i in range(len(values) - time_steps):
        output.append(values[i : (i + time_steps)])
        
    return np.stack(output)

def get_model(x_train):
    model = keras.Sequential(
        [
            layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
            layers.Conv1D(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1D(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1DTranspose(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model

def train_model(model, x_train):
    history = model.fit(
        x_train,
        x_train,
        epochs=60,
        batch_size=128,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
        ],
    )
    return history


models = []
histories = []
for key in dfs:
    df = dfs[key]
    for name in list(df['name'].unique()):
        df_n = df[df.name == name]
        if df_n.shape[0] >30000:
            time_step = 288
        elif df_n.shape[0] > 5000:
            time_step = 144
        else:
            time_step = 12
        x_train=np.empty((1,time_step,1))
        for host in list(df_n['cmdb_id'].unique()):
            df_nh = df_n[df_n.cmdb_id==host][['value']]
            df_nh = normalise(df_nh)
            x_train = np.concatenate((x_train, gen_train_seq(df_nh.values, time_step)), axis=0)

        nan_in = np.isnan(x_train)
        x_train[nan_in] = 0

        print('Training ', name)
        model = get_model(x_train)
        history = train_model(model, x_train)

     
        model.save(save_dir + str(key) + '_' + name)
        models.append(model)
        histories.append(history)




