import pandas as pd
import numpy as np
import xgboost as xgb

# Idea is to do a blurry fit on normal data, then count large diff to prediction as anomalies

test_data = pd.read_csv('/Users/baconbaker/Documents/Studium/ANM/anm-project/data/test_data/esb.csv')
train_data = pd.read_csv('/Users/baconbaker/Documents/Studium/ANM/anm-project/data/train_data/esb.csv')

print(test_data.head())
print(train_data.head())

def get_onehot(df):
    df['start_time'] = pd.to_datetime(df['start_time'], unit='ms', origin='unix')
    df['start_hour'] = df["start_time"].dt.strftime("%H")
    df['start_minute'] = df["start_time"].dt.strftime("%M")
    return df

def split(df):
    df_x = df[['start_hour', 'start_minute']].astype('int32')
    df_y = df[['num', 'avg_time']]
    return df_x, df_y

train_data = get_onehot(train_data)
train_x, train_y = split(train_data)

test_data = get_onehot(test_data)
test_x, test_y = split(test_data)

dtrain = xgb.DMatrix(train_x, label=train_y[['avg_time']])
dtest = xgb.DMatrix(test_x, label=test_y[['avg_time']])

param = {
    'max_depth': 3,
    'eta': 1.6487,
    'objective': 'reg:squaredlogerror'}

param['eval_metric'] = 'rmsle'
evallist = [(dtest, 'eval'), (dtrain, 'train')]
progress = {}

num_round = 60
model = xgb.train(param, dtrain, num_round, evallist, evals_result=progress)
print(progress['eval']['rmsle'][-1])