""" This file provides default estimators and hyperparameters for running cross validation
checks. The file is similar to the 'Est_Params.py'-file except that the default values here are
in []-brackets that are needed if 'GridSearchCV()' or other cross validation functions are run.
Please note: To change single parameter values, change them in the file 'Est_Params.py'-file !!!
To create a list for best Parameter Search in 'GridSearchCV', just add them here within the list brackets !!!"""

from Est_Params import *

cv_est_params = {}

# Gaussian Naive Bayes:
gnb = {'priors': [gnb.get('priors')],
       'var_smoothing': [gnb.get('var_smoothing')]}

# Logistic Regression:
logit = {'C': [logit.get('C')],  # High C -> High Variance, Low C -> High Bias
         'class_weight': [logit.get('class_weight')],
         'dual': [logit.get('dual')],
         'fit_intercept': [logit.get('fit_intercept')],
         'intercept_scaling': [logit.get('intercept_scaling')],
         'l1_ratio': [logit.get('l1_ratio')],
         'max_iter': [logit.get('max_iter')],  # Maximum number of iterations taken for the solvers to converge
         'multi_class': [logit.get('multi_class')],  # For other than binary classes
         'n_jobs': [logit.get('n_jobs')],
         'penalty': [logit.get('penalty')],
         'random_state': [logit.get('random_state')],
         'solver': [logit.get('solver')],
         'tol': [logit.get('tol')],
         'verbose': [logit.get('verbose')],
         'warm_start': [logit.get('warm_start')]}


# Decision Tree Regressor
dtr = {'ccp_alpha': [dtr.get('ccp_alpha')],
       'criterion': [dtr.get('criterion')],
       'max_depth': [dtr.get('max_depth')],
       'max_features': [dtr.get('max_features')],
       'max_leaf_nodes': [dtr.get('max_leaf_nodes')],
       'min_impurity_decrease': [dtr.get('min_impurity_decrease')],
       'min_impurity_split': [dtr.get('min_impurity_split')],
       'min_samples_leaf': [dtr.get('min_samples_leaf')],
       'min_samples_split': [dtr.get('min_samples_split')],
       'min_weight_fraction_leaf': [dtr.get('min_weight_fraction_leaf')],
       'presort': [dtr.get('presort')],
       'random_state': [dtr.get('random_state')],
       'splitter': [dtr.get('splitter')]}

# Random Forest Classifier:
rfc = {'bootstrap': [rfc.get('bootstrap')],
       'ccp_alpha': [rfc.get('ccp_alpha')],
       'class_weight': [rfc.get('class_weight')],
       'criterion': [rfc.get('criterion')],  # Alternative: 'entropy'
       'max_depth': [rfc.get('max_depth')],  # depth of tree: integer
       'max_features': [rfc.get('max_features')],  # maximum number of features that are evaluated for splitting each node
       'max_leaf_nodes': [rfc.get('max_leaf_nodes')],  # maximum number of samples a leaf node can have
       'max_samples': [rfc.get('max_samples')],
       'min_impurity_decrease': [rfc.get('min_impurity_decrease')],
       'min_impurity_split': [rfc.get('min_impurity_split')],
       'min_samples_leaf': [rfc.get('min_samples_leaf')],  # minimum number of samples a leaf node must have
       'min_samples_split': [rfc.get('min_samples_split')],  # minimum number of samples a node must have before it can split
       'min_weight_fraction_leaf': [rfc.get('min_weight_fraction_leaf')],  # same as 'min_samples_leaf' but expressed as fraction of the total
       # number of weighted instances
       'n_estimators': [rfc.get('n_estimators')],  # Number of trees
       'n_jobs': [rfc.get('n_jobs')],
       'oob_score': [rfc.get('oob_score')],
       'random_state': [rfc.get('random_state')],
       'verbose': [rfc.get('verbose')],
       'warm_start': [rfc.get('warm_start')]}

# Support Vector Classifier
svc = {'C': [svc.get('C')],
       'break_ties': [svc.get('break_ties')],
       'cache_size': [svc.get('cache_size')],
       'class_weight': [svc.get('class_weight')],
       'coef0': [svc.get('coef0')],
       'decision_function_shape': [svc.get('decision_function_shape')],
       'degree': [svc.get('degree')],
       'gamma': [svc.get('gamma')],
       'kernel': [svc.get('kernel')],
       'max_iter': [svc.get('max_iter')],
       'probability': [svc.get('probability')],  # If set to TRUE, predict_proba method is available. Otherwise not.
       'random_state': [svc.get('random_state')],
       'shrinking': [svc.get('shrinking')],
       'tol': [svc.get('tol')],
       'verbose': [svc.get('verbose')]}

# XGBoost
xgb = {'objective': [xgb.get('objective')],
       'base_score': [xgb.get('base_score')],
       'booster': [xgb.get('booster')],
       'colsample_bylevel': [xgb.get('colsample_bylevel')],
       'colsample_bynode': [xgb.get('colsample_bynode')],
       'colsample_bytree': [xgb.get('colsample_bytree')],
       'gamma': [xgb.get('gamma')],
       'gpu_id': [xgb.get('gpu_id')],
       'importance_type': [xgb.get('importance_type')],
       'interaction_constraints': [xgb.get('interaction_constraints')],
       'learning_rate': [xgb.get('learning_rate')],  # value between 0 and 1. 1->High Variance, 0->High Bias
       'max_delta_step': [xgb.get('max_delta_step')],
       'max_depth': [xgb.get('max_depth')],
       'min_child_weight': [xgb.get('min_child_weight')],
       'missing': [xgb.get('missing')],
       'monotone_constraints': [xgb.get('monotone_constraints')],
       'n_estimators': [xgb.get('n_estimators')],  # Number of trees: Each new tree regresses residuals of last
       # tree only and adds prediction values to previous predictions
       'n_jobs': [xgb.get('n_jobs')],
       'num_parallel_tree': [xgb.get('num_parallel_tree')],
       'random_state': [xgb.get('random_state')],
       'reg_alpha': [xgb.get('reg_alpha')],
       'reg_lambda': [xgb.get('reg_lambda')],
       'scale_pos_weight': [xgb.get('scale_pos_weight')],
       'subsample': [xgb.get('subsample')],  # Enables Stochastic XGB: value between 0 and 1. For instance,
       # subsample = 0.25, then only 25% of the training set are used for training each tree
       'tree_method': [xgb.get('tree_method')],
       'validate_parameters': [xgb.get('validate_parameters')],
       'verbosity': [xgb.get('verbosity')]}

# AdaBoostClassifier
abc = {'algorithm': [abc.get('algorithm')],
       'base_estimator': [abc.get('base_estimator')],
       'learning_rate': [abc.get('learning_rate')],
       'n_estimators': [abc.get('n_estimators')],
       'random_state': [abc.get('random_state')]}

#cv_est_params.update({'gnb': {'estimator': GaussianNB(), 'individual': False, 'hyper_param': gnb}})
cv_est_params.update({'logit': {'estimator': LogisticRegression(), 'individual': False, 'hyper_param': logit}})
#cv_est_params.update({'dtr': {'estimator': DecisionTreeRegressor(), 'individual': True, 'hyper_param': dtr}})
cv_est_params.update({'rfc': {'estimator': RandomForestClassifier(), 'individual': False, 'hyper_param': rfc}})
#cv_est_params.update({'svc': {'estimator': SVC(), 'individual': False, 'hyper_param': svc}})
cv_est_params.update({'xgb': {'estimator': XGBClassifier(), 'individual': False, 'hyper_param': xgb}})
#cv_est_params.update({'abc': {'estimator': AdaBoostClassifier(), 'individual': False, 'hyper_param': abc}})
