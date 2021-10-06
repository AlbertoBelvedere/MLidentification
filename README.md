# MLidentification

This directory contains the code necessary to train and evaluate the performance of the Machine Learning algorithm for the low-pt identification for the B-parking dataset reprocessing.

The training is performed using **train_bdt.py** which implements the XGBoost algorithm, trough **dataset.py** it is possible to choose the dataset and its charateristics, while using **features.py** one can choose the list of features to use in the training. First of all it's necessary to run **kmeans_reweight.py** to compute the weights to eliminate discrepancies between the distributions of the kinematic variables of electrons and fakes. Then, to train with a specific set of features:

```
python train_bdt.py list_of_features
```

where list_of_features is a list of features presents in **features.py** .<br/>
While to run without using weights:

```
python train_bdt.py list_of_features --noweight
```

**accuracy.py** : algorithm's accuracy computation.<br/>
**basic_plots.py** : performance of the algorithm on the old dataset.<br/>
**compare_...** : these files compare the performance of different types of training.<br/>
**correlation_matrix.py** : correlation matrices computation.<br/>
**datasets.py** : choice of the dataset and its carachteristic.<br/>
**eval_bdt.py** : evaluation of the algorithm performance trough the analysis of the ROC curve.<br/>
**features.py** : lists of features that can be used to train the algorithm.<br/>
**feature_imortance.py** : feature importance plot.<br/>
**info_parameters.py** : run to get the parameters of the model.<br/>
**kmeans_reweight.py** : reweight of the kinematical variables.<br/>
**mistag_rate.py** : mistag rate and efficiency computation.<br/>
**train_bdt.py** : file to train the algorithm.
