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
    'models/2020Nov28ULALL/bdt_cmssw_mva_id_nnclean2_forUL/'
    '/bdt_cmssw_mva_id_nnclean2_forUL_testdata.hdf', key='data')
test = test[np.invert(test.is_egamma)] 
test = test[np.invert(abs(test.gsf_mode_eta)>=2.4)] 
test = test[np.invert(test.gsf_mode_pt<0.5)] 

test1 = pd.read_hdf(
    'models/2020Nov28ULALL/bdt_retrain/'
    '/bdt_retrain_testdata.hdf', key='data')
test1 = test1[np.invert(test1.is_egamma)] 
test1 = test1[np.invert(abs(test1.gsf_mode_eta)>=2.4)] 
test1 = test1[np.invert(test1.gsf_mode_pt<0.5)]

test2 = pd.read_hdf(
    'models/2020Nov28ULALL/bdt_retrain_final/'
    '/bdt_retrain_final_testdata.hdf', key='data')
test2 = test[np.invert(test2.is_egamma)] 
test2 = test[np.invert(abs(test2.gsf_mode_eta)>=2.4)] 
test2 = test[np.invert(test2.gsf_mode_pt<0.5)]

test3 = pd.read_hdf(
    'models/2020Nov28ULALL/bdt_final_train_11/'
    '/bdt_final_train_11_testdata.hdf', key='data')
test3 = test[np.invert(test3.is_egamma)] 
test3 = test[np.invert(abs(test3.gsf_mode_eta)>=2.4)] 
test3 = test[np.invert(test3.gsf_mode_pt<0.5)]
#print test.size

# default variables
cmssw_mva_id_nnclean2_forUL = get_model(
    'models/2020Nov28ULALL/bdt_cmssw_mva_id_nnclean2_forUL/'
    '/2020Nov28ULALL__cmssw_mva_id_nnclean2_forUL_BDT.pkl')
cmssw_mva_id_nnclean2_forUL_features, _ = get_features('cmssw_mva_id_nnclean2_forUL')
test['cmssw_mva_id_nnclean2_forUL_out'] = cmssw_mva_id_nnclean2_forUL.predict_proba(test[cmssw_mva_id_nnclean2_forUL_features].as_matrix())[:,1]
test['cmssw_mva_id_nnclean2_forUL_out'].loc[np.isnan(test.cmssw_mva_id_nnclean2_forUL_out)] = -999 
cmssw_mva_id_nnclean2_forUL_roc = roc_curve(
    test.is_e, test.cmssw_mva_id_nnclean2_forUL_out
    )
cmssw_mva_id_nnclean2_forUL_auc = roc_auc_score(test.is_e, test.cmssw_mva_id_nnclean2_forUL_out)
print "ROC done"
print cmssw_mva_id_nnclean2_forUL_auc

retrain = get_model(
    'models/2020Nov28ULALL/bdt_retrain/'
    '/2020Nov28ULALL__retrain_BDT.pkl')
retrain_features, _ = get_features('retrain')
test1['retrain_out'] = retrain.predict_proba(test1[retrain_features].as_matrix())[:,1]
test1['retrain_out'].loc[np.isnan(test1.retrain_out)] = -999 
retrain_roc = roc_curve(
    test1.is_e, test1.retrain_out
    )
retrain_auc = roc_auc_score(test1.is_e, test1.retrain_out)
print "ROC done"
print retrain_auc


retrain_final = get_model(
    'models/2020Nov28ULALL/bdt_retrain_final/'
    '/2020Nov28ULALL__retrain_final_BDT.pkl')
retrain_final_features, _ = get_features('retrain_final')
test2['retrain_final_out'] = retrain_final.predict_proba(test2[retrain_final_features].as_matrix())[:,1]
test2['retrain_final_out'].loc[np.isnan(test2.retrain_final_out)] = -999 
retrain_final_roc = roc_curve(
    test2.is_e, test2.retrain_final_out
    )
retrain_final_auc = roc_auc_score(test2.is_e, test2.retrain_final_out)
print "ROC done"
print retrain_final_auc

final_train_11 = get_model(
    'models/2020Nov28ULALL/bdt_final_train_11/'
    '/2020Nov28ULALL__final_train_11_BDT.pkl')
final_train_11_features, _ = get_features('final_train_11')
test3['final_train_11_out'] = final_train_11.predict_proba(test3[final_train_11_features].as_matrix())[:,1]
test3['final_train_11_out'].loc[np.isnan(test3.final_train_11_out)] = -999 
final_train_11_roc = roc_curve(
    test3.is_e, test3.final_train_11_out
    )
final_train_11_auc = roc_auc_score(test3.is_e, test3.final_train_11_out)
print "ROC done"
print final_train_11_auc

retrain_final_5 = get_model(
    'models/2020Nov28ULALL/modello/bdt_retrain_final_5/'
    '/2020Nov28ULALL__retrain_final_5_BDT.pkl')
retrain_final_5_features, _ = get_features('retrain_final_5')
test3['retrain_final_5_out'] = retrain_final_5.predict_proba(test3[retrain_final_5_features].as_matrix())[:,1]
test3['retrain_final_5_out'].loc[np.isnan(test3.retrain_final_5_out)] = -999 
retrain_final_5_roc = roc_curve(
    test3.is_e, test3.retrain_final_5_out
    )
retrain_final_5_auc = roc_auc_score(test3.is_e, test3.retrain_final_5_out)
print "ROC done"
print retrain_final_5_auc

## training version in cmssw
cmssw_roc = roc_curve(
     test.is_e, test.ele_mva_value
    )
cmssw_auc = roc_auc_score(test.is_e, test.ele_mva_value)
print cmssw_auc

# plots
print "Making plots ..."

# ROCs
plt.figure(figsize=[8, 12])
ax = plt.subplot(111)  
box = ax.get_position()   
ax.set_position([box.x0, box.y0, box.width, box.height*0.666]) 

plt.title('Trainings comparison')
plt.plot(
   np.arange(0,1,0.01),
   np.arange(0,1,0.01),
   'k--')


#plt.plot(cmssw_mva_id_nnclean2_forUL_roc[0][:-1], cmssw_mva_id_nnclean2_forUL_roc[1][:-1], 
         #linestyle='dashed', 
         #color='black', 
         #label='UL retraining (AUC: %.3f)' %cmssw_mva_id_nnclean2_forUL_auc)

#plt.plot(retrain_final_5_roc[0][:-1], retrain_final_5_roc[1][:-1], 
#         linestyle='dashed', 
#         color='red', 
#         label='Final train (AUC: %.3f)' %retrain_final_5_auc)

mva_v2 = roc_curve(test.is_e, test.ele_mva_value)[:2]
mva_v2_auc = roc_auc_score(test.is_e, test.ele_mva_value)
#rocs['mva_v2'] = mva_v2
plt.plot(*mva_v2, label='RK ID (Sept 15) (AUC: %.3f)'  % mva_v2_auc, color = 'blue')

plt.plot(final_train_11_roc[0][:-1], final_train_11_roc[1][:-1], 
         #linestyle='dashed', 
         color='red', 
         label='Final train (AUC: %.3f)' %final_train_11_auc)

#plt.plot(retrain_roc[0][:-1], retrain_roc[1][:-1], 
#         linestyle='dashed', 
#         color='red', 
#         label='No correlated (AUC: %.3f)' %retrain_auc)

#plt.plot(retrain_final_roc[0][:-1], retrain_final_roc[1][:-1], 
#         linestyle='dashed', 
#         color='blue', 
#         label='No low importance (AUC: %.3f)' %retrain_final_auc)

plt.xlabel('Mistag Rate')
plt.ylabel('Efficiency')
plt.legend(loc='best')
plt.xlim(0., 1)
plt.ylim(0., 1.05)
plt.savefig('ROC_comparison5.png')
plt.gca().set_xscale('log')
plt.ylim(0., 1.05)
plt.xlim(1e-4, 1)
plt.savefig('ROC_comparison_log5.png')
plt.clf()





