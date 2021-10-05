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

print test.size

# default variables
base = get_model(
    'models/2020Nov28ULALL/bdt_cmssw_mva_id_nnclean2_forUL/'
    '/2020Nov28ULALL__cmssw_mva_id_nnclean2_forUL_BDT.pkl')
based_features, _ = get_features('cmssw_mva_id_nnclean2_forUL')
test['base_out'] = base.predict_proba(test[based_features].as_matrix())[:,1]
test['base_out'].loc[np.isnan(test.base_out)] = -999 
base_roc = roc_curve(
    test.is_e, test.base_out
    )
base_auc = roc_auc_score(test.is_e, test.base_out)
print "ROC done"
print base_auc
cmssw_roc = roc_curve(
    test.is_e, test.ele_mva_value
    )
cmssw_auc = roc_auc_score(test.is_e, test.ele_mva_value)
#cmssw_roc = roc_curve(validation.is_e, validation.ele_mva_value)[:2]
#cmssw_auc = roc_auc_score(validation.is_e, validation.ele_mva_value)
print cmssw_auc

# plots
print "Making plots ..."


# 1dim distribution
plt.title('BDT output')
basesignal = test.base_out.as_matrix()
basesignal = basesignal[test.is_e==1]
basebkg = test.base_out.as_matrix()
basebkg = basebkg[test.is_e==0]
plt.hist(basesignal, bins=70, color="green", lw=0, label='signal',normed=1,alpha=0.5)
plt.hist(basebkg, bins=70, color="skyblue", lw=0, label='bkg',normed=1,alpha=0.5)
plt.show()
plt.legend(loc='best')
plt.savefig('OUTBase_comparison.png')
plt.clf()

# some working points
print ''
jmap = {}
for base_thr, wpname in [
        #    (1.83 , 2.61, 'T'),
#    (0.76 , 1.75, 'M'),
#    (-0.48, 1.03, 'L'),
#    (1.45 , 2.61, 'T+'),
#    (0.33 , 1.75, 'M+'),
#    (-0.97, 1.03, 'L+'),
#    ]:
#    (1.83 , 1.83, 'T'),
#    (0.60 , 0.60, 'M'),
#    (-0.56, -0.56, 'L'),
#    (5. , 5., 'T1'),
   (5. , 'T1'),
    (4.8 , 'T2'),
    (4.6 , 'T3'),
    (4.4 , 'T4'),
    (4.2 , 'T5'),
    (4.0 , 'T6'),
    (3.8 , 'T7'),
    (3.6 , 'T8'),
    (3.4 , 'T9'),
    (3.2 , 'T10'),
    (3.0 , 'T11'),
    (2.8 , 'T12'),
    (2.6 , 'T13'),
    (2.55 , 'T14'),
    (2.50 , 'T15'),
    (2.45 , 'T16'),
    (2.4 , 'T17'),
    (2.35 , 'T18'),
    (2.3 , 'T13'),
    (2.25 , 'T14'),
    (2.21 , 'T1*'),
    (2.22 , 'T2*'),
    (2.23 , 'T3*'),
    (2.24 , 'T4*'),
    (2.2 , 'T15'),
    (2.19 , 'T1**'),
    (2.18 , 'T2**'),
    (2.17 , 'T3**'),
    (2.16 , 'T4**'),
    (2.15 , 'T16'),
    (2.14 , 'T1****'),
    (2.135 , 'T1*****'),
    (2.13 , 'T2****'),
    (2.12 , 'T3****'),
    (2.11 , 'T4****'),
    (2.1 , 'T17'),
    (2.09 , 'T1***'),
    (2.08 , 'T2***'),
    (2.07 , 'T3***'),
    (2.06 , 'T4***'),
    (2.05, 'T18'),
    (2.04 , 'T1**'),
    (2.03 , 'T2**'),
    (2.02 , 'T3**'),
    (2.01 , 'T4**'),
    (2. , 'T13'),
    (1.95 , 'T14'),
    (1.94 , 'T1**'),
    (1.93 , 'T2**'),
    (1.92 , 'T3**'),
    (1.91 , 'T4**'),
    (1.9 , 'T15'),
    (1.85 , 'T16'),
    (1.8 , 'T18'),
    (1.7 , 'T19'),
    (1.6 , 'T20'),
    (1.5 , 'T21'),
    (1.4 , 'T22'),
    (1.3 , 'T23'),
    (1.2 , 'T24'),
    (1.1 , 'T25'),
    (1.0 , 'T26'),
    (0.9 , 'T27'),
    (0.8 , 'T28'),
    (0.7 , 'T29'),
    (0.6 , 'T30'),
    (0.5 , 'T31'),
    (0.4 , 'T32'),
    (0.3 , 'T33'),
    (0.2 , 'T34'),
    (0.1 , 'T35'),
    (0. ,  'T36'),
    (-0.1 , 'T37'),
    (-0.2 , 'T38'),
    (-0.3 , 'T39'),
    (-0.4 , 'T40'),
    (-0.5 , 'T41'),
    (-0.6 , 'T42'),
    (-0.7 , 'T43'),
    (-0.8 , 'T44'),
    (-0.9 , 'T45'),
    (-1. ,  'T46'),
    ]:
   print 'WP', wpname
   print 'base:'
   test['base_pass'] = test.base_out > base_thr
#   print 'ecal:'
#   test['ecal_pass'] = test.ecal_out > ecal_thr
    
   eff_base = ((test.base_pass & test.is_e).sum()/float(test.is_e.sum()))
   mistag_base = ((test.base_pass & np.invert(test.is_e)).sum()/float(np.invert(test.is_e).sum()))
#   eff_ecal = ((test.ecal_pass & test.is_e).sum()/float(test.is_e.sum()))
#   mistag_ecal = ((test.ecal_pass & np.invert(test.is_e)).sum()/float(np.invert(test.is_e).sum()))

   jmap[wpname] = [mistag_base, eff_base]
   print 'eff (base): %.5f' % eff_base
   print 'mistag (base): %.5f' % mistag_base
#   print 'eff (ecal): %.3f' % eff_ecal
#   print 'mistag (ecal): %.3f' % mistag_ecal
