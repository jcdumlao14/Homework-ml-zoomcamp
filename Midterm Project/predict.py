import pickle

import xgboost as xgb

from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model.bin'

with open(model_file,'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('insuranceclaim')

@app.route('/predict', methods =['POST'])
def predict():
    customer = request.get_json()
    
    X = dv.transform([customer])
    dtest = xgb.DMatrix(X, feature_names=dv.get_feature_names_out())
    y_pred = model.predict(dtest)
    stratification = y_pred > 0.3
    
    result = {
    'insuranceclaim_probability': float(y_pred),
    'insuranceclaim': bool(stratification)
    }
    
    return jsonify(result) 