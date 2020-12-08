import numpy as np
import pandas as pd
import os

from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

from sklearn import preprocessing

data_path = '../../../data/train_data/host'
save_dir = '' # Model save-file directory
dfs = {}
   
for file in os.listdir(data_path):
    print('Saving ' + file[:-4] + ' into dfs')
    dfs[file[:-4]] = pd.read_csv(data_path+'/'+file) 

def normalise(df):
    df['value'] = preprocessing.scale(df['value'].values)
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
        epochs=120,
        batch_size=128,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
        ],
    )
    return history

models = []
failures = []
histories = []
cancel = False
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
        x_train_list=[]
        for host in list(df_n['cmdb_id'].unique()):
            df_nh = df_n[df_n.cmdb_id==host][['value']]
            if len(df_nh.values) - time_step <= 0:
              failures.append((name, host))
              cancel = True
            else:
              df_nh = normalise(df_nh)
              x_train_list.append(gen_train_seq(df_nh.values, time_step))

        if cancel:
          print("Cancelling for ", failures[-1])
          cancel = False
          continue
        xt = np.concatenate(x_train_list)
        print(xt.shape)
        model = get_model(xt)
        history = train_model(model, xt)
     
        model.save(save_dir + str(key) + '_' + name)
        models.append(model)
        histories.append(history)