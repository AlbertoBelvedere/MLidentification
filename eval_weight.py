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
import seaborn as sns


def get_model(pkl):
    model = joblib.load(pkl)

    def _monkey_patch():
        return model._Booster

    if isinstance(model.booster, basestring):
        model.booster = _monkey_patch
    return model

# test dataset
test = pd.read_hdf(
    'models/2020Nov28ULALL/bdt_retrain_final/'
    '/bdt_retrain_final_testdata.hdf', key='data')


test = test[np.invert(test.is_egamma)] 
test = test[np.invert(abs(test.gsf_mode_eta)>=2.4)] 
test = test[np.invert(test.gsf_mode_pt<0.5)] 



print test.size

# default variables
base = get_model(
    'models/2020Nov28ULALL/bdt_retrain_final/'
    '2020Nov28ULALL__retrain_final_BDT.pkl')
based_features, _ = get_features('retrain_final')

# plots
print "Making plots ..."


plt.figure(figsize=[8, 12])
ax = plt.subplot(111)  
box = ax.get_position()   
ax.set_position([box.x0, box.y0, box.width, box.height*0.666]) 

plt.title('Weight')
basesignal = test.weight[test.is_e==1]
basebkg = test.weight[test.is_e==0]
plt.hist(basesignal, bins=70, color="green", lw=0, label='signal',normed=1,alpha=0.5)
plt.hist(basebkg, bins=70, color="skyblue", lw=0, label='bkg',normed=1,alpha=0.5)
plt.show()
plt.legend(loc='best')
plt.savefig('generico/Weight.png')
plt.clf()

plt.title('Weight background')
plt.hist(basebkg, bins=70, color="skyblue", lw=0, label='bkg',normed=1,alpha=0.5)
plt.show()
plt.legend(loc='best')
plt.savefig('generico/WeightBackground.png')
plt.clf()



drop = [
    'ele_eta',
    'ele_mva_id',
    'ele_mva_value',
    'ele_pt',
    'evt',
    'gen_dR',
    'gen_eta',
    'gen_pt',
    'gsf_eta',
    'gsf_pt',
    'has_ele',
    'has_gsf',
    'has_seed',
    'has_trk',
    'is_e',
    'is_e_not_matched',
    'is_egamma',
    'is_other',
    'sc_energy',
    'sc_raw_energy',
    'weight',
    'prescale',
    'training_out',
    'log_gsfmodept',
]

feats = test.drop(columns=drop)
feats_elec = feats[test.is_e == 1]
feats_fakes = feats[test.is_e == 0]


#electrons

corrmatr = feats_elec.corr().abs()
tfile = open('generico/correlazione_elec.txt', 'w')
tfile.write(corrmatr.to_string())
tfile.close()

plt.figure(figsize=[15, 12])
sns.heatmap(corrmatr, cmap="YlGnBu")
plt.title("Correlation heatmap electrons")
plt.subplots_adjust(left = 0.22, bottom = 0.18, right = 0.99, top = 0.85, wspace = 0, hspace =0)
plt.show()
plt.savefig('generico/Corr_heatmap_1_elec.png')
plt.clf()

threshold = 0.5
corrmatr_mask = corrmatr
corrmatr_mask[corrmatr_mask < threshold] = 0
Ar_annotation = corrmatr_mask.as_matrix()
Ar_annotation[Ar_annotation == 0] = None

plt.figure(figsize=[15, 12])
sns.heatmap(corrmatr,  cmap="YlGnBu", annot = Ar_annotation, fmt = ".2f", annot_kws={"fontsize":9} )
plt.title("Correlation heatmap > 0.5 electrons")
plt.subplots_adjust(left = 0.22, bottom = 0.18, right = 0.99, top = 0.85, wspace = 0, hspace =0)
plt.show()
plt.savefig('generico/Corr_heatmap_2_elec.png')
plt.clf()



#fakes

corrmatr = feats_fakes.corr().abs()
tfile = open('generico/correlazione_fakes.txt', 'w')
tfile.write(corrmatr.to_string())
tfile.close()

plt.figure(figsize=[15, 12])
sns.heatmap(corrmatr, cmap="YlGnBu")
plt.title("Correlation heatmap fakes")
plt.subplots_adjust(left = 0.22, bottom = 0.18, right = 0.99, top = 0.85, wspace = 0, hspace =0)
plt.show()
plt.savefig('generico/Corr_heatmap_1_fakes.png')
plt.clf()

threshold = 0.5
corrmatr_mask = corrmatr
corrmatr_mask[corrmatr_mask < threshold] = 0
Ar_annotation = corrmatr_mask.as_matrix()
Ar_annotation[Ar_annotation == 0] = None

plt.figure(figsize=[15, 12])
sns.heatmap(corrmatr,  cmap="YlGnBu", annot = Ar_annotation, fmt = ".2f", annot_kws={"fontsize":9} )
plt.title("Correlation heatmap > 0.5 fakes")
plt.subplots_adjust(left = 0.22, bottom = 0.18, right = 0.99, top = 0.85, wspace = 0, hspace =0)
plt.show()
plt.savefig('generico/Corr_heatmap_2_fakes.png')
plt.clf()
