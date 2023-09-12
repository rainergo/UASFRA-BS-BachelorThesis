""" This file provides default estimators and hyperparameters that can be passed to a function.
The file is similar to the 'CV_Est_Params.py'-file except that the default values here are
not in []-brackets as otherwise they would not be accepted as function parameters."""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

est_params = {}

# Gaussian Naive Bayes:
gnb = {'priors': None,
       'var_smoothing': 1e-09}

# Logistic Regression:
logit = {'C': 0.1,  # High C -> High Variance, Low C -> High Bias
         'class_weight': None,
         'dual': False,
         'fit_intercept': True,
         'intercept_scaling': 1,
         'l1_ratio': None,
         'max_iter': 100,  # Maximum number of iterations taken for the solvers to converge
         'multi_class': 'auto',  # For other than binary classes
         'n_jobs': None,
         'penalty': 'l2',
         'random_state': None,
         'solver': 'lbfgs',
         'tol': 0.0001,
         'verbose': 0,
         'warm_start': False}

# Decision Tree Regressor
dtr = {'ccp_alpha': 0.0,
       'criterion': 'mse',
       'max_depth': None,
       'max_features': None,
       'max_leaf_nodes': None,
       'min_impurity_decrease': 0.0,
       'min_impurity_split': None,
       'min_samples_leaf': 1,
       'min_samples_split': 2,
       'min_weight_fraction_leaf': 0.0,
       'presort': 'deprecated',
       'random_state': None,
       'splitter': 'best'}


# Random Forest Classifier:
rfc = {'bootstrap': True,
       'ccp_alpha': 0.0,
       'class_weight': None,
       'criterion': 'gini',  # Alternative: 'entropy'
       'max_depth': None,  # depth of tree: integer
       'max_features': 'log2',  # maximum number of features that are evaluated for splitting each node
       'max_leaf_nodes': None,  # maximum number of samples a leaf node can have
       'max_samples': None,
       'min_impurity_decrease': 0.0,
       'min_impurity_split': None,
       'min_samples_leaf': 10,  # minimum number of samples a leaf node must have
       'min_samples_split': 20,  # minimum number of samples a node must have before it can split
       'min_weight_fraction_leaf': 0.0,  # same as 'min_samples_leaf' but expressed as fraction of the total
       # number of weightd instances
       'n_estimators': 100,  # Number of trees
       'n_jobs': None,
       'oob_score': False,
       'random_state': None,
       'verbose': 0,
       'warm_start': False}

# Support Vector Classifier
svc = {'C': 1.0,
       'break_ties': False,
       'cache_size': 200,
       'class_weight': None,
       'coef0': 0.0,
       'decision_function_shape': 'ovr',
       'degree': 3,
       'gamma': 'scale',
       'kernel': 'linear',
       'max_iter': -1,
       'probability': True,  # If set to TRUE, predict_proba method is available. Otherwise not.
       'random_state': 49,
       'shrinking': True,
       'tol': 0.001,
       'verbose': False}

# XGBoost
xgb = {'objective': 'binary:logistic',
       'base_score': None,
       'booster': None,
       'colsample_bylevel': None,
       'colsample_bynode': None,
       'colsample_bytree': None,  # Default: None, else 0.5, 0.7. Higher value -> Higher variance
       'gamma': None,  # Default: None, 0, 0.5, 2.0, 10.0, Higher value -> Lower variance
       'gpu_id': None,
       'importance_type': 'gain',
       'interaction_constraints': None,
       'learning_rate': None,  # Default: None, else between 0 and 1. 1->High Variance, 0->High Bias:
       # 0.01, 0.05, 0.1, 0.2, 0.5, 1.0 Higher value -> Higher variance
       'max_delta_step': None,
       'max_depth': None,  # Default: None, else 3, 4, 5, 7, 10. Higher value -> Higher variance
       'min_child_weight': 10,  # Default: 1, else 3, 5. Higher value -> Lower variance
       'missing': np.nan,
       'monotone_constraints': None,
       'n_estimators': 100,  # Number of trees: Each new tree regresses residuals of last
       # tree only and adds prediction values to previous predictions. Higher value -> Higher variance
       'n_jobs': None,
       'num_parallel_tree': None,
       'random_state': None,
       'reg_alpha': None,
       'reg_lambda': None,
       'scale_pos_weight': None,
       'subsample': None,  # Enables Stochastic XGB: value between 0 and 1. For instance,
       # subsample = 0.25, then only 25% of the training set are used
       # for training each tree. Higher value -> Lower variance
       'tree_method': None,
       'validate_parameters': None,
       'verbosity': None}

xgb_strict = {'objective': 'binary:logistic',
              'base_score': None,
              'booster': None,
              'colsample_bylevel': None,
              'colsample_bynode': None,
              'colsample_bytree': 0.5,  # Default: None, else 0.5, 0.7. Higher value -> Higher variance
              'gamma': 10.0,  # Default: None, 0, 0.5, 2.0, 10.0, Higher value -> Lower variance
              'gpu_id': None,
              'importance_type': 'gain',
              'interaction_constraints': None,
              'learning_rate': 0.001,  # Default: None, else between 0 and 1. 1->High Variance, 0->High Bias:
              # 0.01, 0.05, 0.1, 0.2, 0.5, 1.0 Higher value -> Higher variance
              'max_delta_step': None,
              'max_depth': 3,  # Default: None, else 3, 4, 5, 7, 10. Higher value -> Higher variance
              'min_child_weight': 10,  # Default: None, else 3, 5. Higher value -> Lower variance
              'missing': np.nan,
              'monotone_constraints': None,
              'n_estimators': 10,  # Number of trees: Each new tree regresses residuals of last
              # tree only and adds prediction values to previous predictions. Higher value -> Higher variance
              'n_jobs': None,
              'num_parallel_tree': None,
              'random_state': None,
              'reg_alpha': None,
              'reg_lambda': None,
              'scale_pos_weight': None,
              'subsample': 1.0,  # Enables Stochastic XGB: value between 0 and 1. For instance,
              # subsample = 0.25, then only 25% of the training set are used
              # for training each tree. Higher value -> Lower variance
              'tree_method': None,
              'validate_parameters': None,
              'verbosity': None}

# AdaBoostClassifier
abc = {'algorithm': 'SAMME.R',
       'base_estimator': None,
       'learning_rate': 1.0,
       'n_estimators': 50,
       'random_state': None}

est_params.update({'gnb': {'estimator': GaussianNB(), 'individual': False, 'hyper_param': gnb}})
est_params.update({'logit': {'estimator': LogisticRegression(), 'individual': True, 'hyper_param': logit}})
est_params.update({'dtr': {'estimator': DecisionTreeRegressor(), 'individual': True, 'hyper_param': dtr}})
est_params.update({'rfc': {'estimator': RandomForestClassifier(), 'individual': False, 'hyper_param': rfc}})
est_params.update({'svc': {'estimator': SVC(), 'individual': False, 'hyper_param': svc}})
est_params.update({'xgb': {'estimator': XGBClassifier(), 'individual': False, 'hyper_param': xgb}})
est_params.update({'abc': {'estimator': AdaBoostClassifier(), 'individual': False, 'hyper_param': abc}})
