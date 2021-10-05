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
from sklearn.metrics import roc_curve , roc_auc_score

def get_model(pkl):
    model = joblib.load(pkl)

    def _monkey_patch():
        return model._Booster

    if isinstance(model.booster, basestring):
        model.booster = _monkey_patch
    return model

# test dataset
test = pd.read_hdf(
    'models/2020Nov28ULALL/bdt_retrain_final_5_pt__2_5/'
    '/bdt_retrain_final_5_pt__2_5_testdata.hdf', key='data')
test = test[np.invert(test.is_egamma)] 
test = test[np.invert(abs(test.gsf_mode_eta)>=2.4)] 
test = test[np.invert(test.gsf_mode_pt<0.5)] 
print "test dataset done"
print test.size

# variables
partial = get_model(
    'models/2020Nov28ULALL/bdt_retrain_final_5_pt__2_5/'
    '/2020Nov28ULALL__retrain_final_5_pt__2_5_BDT.pkl')
partial_features, _ = get_features('retrain_final_5_pt_2_5')

# on test
test['partial_out'] = partial.predict_proba(test[partial_features].as_matrix())[:,1]
test['partial_out'].loc[np.isnan(test.partial_out)] = -999 
test_partial_roc = roc_curve(
    test.is_e, test.partial_out
    )
test_partial_auc = roc_auc_score(test.is_e, test.partial_out)
print "test dataset ROC done"
print test_partial_auc


standard = get_model(
    'models/2020Nov28ULALL/bdt_retrain_final_5_cv_hyper2/'
    '/2020Nov28ULALL__retrain_final_5_BDT.pkl')
standard_features, _ = get_features('retrain_final_5_pt_2_5')

# on test
test['standard_out'] = standard.predict_proba(test[standard_features].as_matrix())[:,1]
test['standard_out'].loc[np.isnan(test.standard_out)] = -999 
test_standard_roc = roc_curve(
    test.is_e, test.standard_out
    )
test_standard_auc = roc_auc_score(test.is_e, test.standard_out)
print "test dataset ROC done"
print test_standard_auc

# plots
print "Making plots ..."

print(partial.get_params)
print(test.shape[0])

# ROCs
plt.figure(figsize=[8, 12])
ax = plt.subplot(111)  
box = ax.get_position()   
ax.set_position([box.x0, box.y0, box.width, box.height*0.666]) 

plt.title('Trainings comparison pt>2.5')
plt.plot(
   np.arange(0,1,0.01),
   np.arange(0,1,0.01),
   'k--')

plt.plot(test_standard_roc[0][:-1], test_standard_roc[1][:-1], 
         linestyle='solid', 
         color='green', 
         label='Train full dataset (AUC: %.3f)' %test_standard_auc)

plt.plot(test_partial_roc[0][:-1], test_partial_roc[1][:-1], 
         linestyle='dashed', 
         color='red', 
         label='Train pt>2.5 dataset (AUC: %.3f)' %test_partial_auc)

plt.xlabel('Mistag Rate')
plt.ylabel('Efficiency')
plt.legend(loc='best')
plt.xlim(0., 1)
plt.gca().set_xscale('log')
plt.xlim(1e-4, 1)
plt.savefig('compare_pt__2_5.png')
#plt.clf()
