#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
import numpy as np

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error


# parameters

output_file = 'model.bin'


# Data 

df = pd.read_csv('https://raw.githubusercontent.com/jcdumlao14/Homework-ml-zoomcamp/main/insurance2.csv')

df['smoker'] = df['smoker'].map({0:'non-smoke',1:'smoker'})
df['sex'] = df['sex'].map({0:'female',1:'male'})
df['insuranceclaim'] = df['insuranceclaim'].map({0:'no',1:'yes'})


df.sex = (df.sex == 'male').astype(int)
df.smoker = (df.smoker == 'smoker').astype(int)
df.insuranceclaim = (df.insuranceclaim == 'yes').astype(int)



df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=25)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
y_full_train = df_full_train.insuranceclaim.values



df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


y_train = df_train.insuranceclaim.values
y_val = df_val.insuranceclaim.values
y_test = df_test.insuranceclaim.values

del df_train['insuranceclaim']
del df_val['insuranceclaim']
del df_test['insuranceclaim']


categorical = ['sex', 'smoker','region','children']

numerical = ['age', 'bmi','charges']

features = numerical + categorical
target = ['insuranceclaim']


# training


def train(df_train,y_train):
    dicts = df_train[categorical + numerical].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=dv.get_feature_names_out())
    
    xgb_params = {
    'eta': 1.0, 
    'max_depth': 6,
    'min_child_weight': 30,
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
    }
    
    model = xgb.train(xgb_params, dtrain, num_boost_round=100)
    
    return dv, model


def predict(df,dv,model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    
    X = dv.transform(dicts)
    dtest = xgb.DMatrix(X, feature_names=dv.get_feature_names_out())
    y_pred = model.predict(dtest)
    
    return y_pred


# #training the final model

print('training the final model')


dv,model = train(df_full_train,y_full_train)
y_pred = predict(df_test,dv,model)

auc = roc_auc_score(y_test, y_pred)
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))

print('auc =',auc)
print('rmse =',rmse)


# #Save the model

with open(output_file,'wb') as f_out:
    pickle.dump((dv, model),f_out)

print(f'the model is saved in {output_file}')
