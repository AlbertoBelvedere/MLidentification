import numpy as np
import matplotlib
matplotlib.use('Agg')
from cmsjson import CMSJson
from pdb import set_trace
import os
from glob import glob
import pandas as pd
import json
from pprint import pprint
import matplotlib.pyplot as plt
from features import *
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.externals import joblib
import xgboost as xgb
from datasets import HistWeighter
from itertools import cycle
from sklearn.metrics import roc_curve , roc_auc_score, accuracy_score

def get_model(pkl):
    model = joblib.load(pkl)

    def _monkey_patch():
        return model._Booster

    if isinstance(model.booster, basestring):
        model.booster = _monkey_patch
    return model

# test dataset
train = pd.read_hdf(
    'models/2020Nov28ULALL/bdt_retrain_final_7/'
    '/bdt_retrain_final_7_traindata.hdf', key='data')
train = train[np.invert(train.is_egamma)] 
train = train[np.invert(abs(train.gsf_mode_eta)>=2.4)] 
train = train[np.invert(train.gsf_mode_pt<0.5)] 

print train.size

# default variables
base = get_model(
    'models/2020Nov28ULALL/bdt_retrain_final_7/'
    '/2020Nov28ULALL__retrain_final_7_BDT.pkl')
based_features, _ = get_features('retrain_final_7')
train['base_out'] = base.predict_proba(train[based_features].as_matrix())[:,1]
train_predict = base.predict(train[based_features].as_matrix())
train['base_out'].loc[np.isnan(train.base_out)] = -999 


#acc = accuracy_score(train.is_e, train.ele_mva_value)
acc = accuracy_score(train.is_e, train_predict)
print(acc)
