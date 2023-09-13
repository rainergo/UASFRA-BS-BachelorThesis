import pandas as pd
import numpy as np
from typing import List
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, GroupKFold
from imblearn.under_sampling import RandomUnderSampler

from modelling.functions import check_parameter_is_of_type, make_param_grid_for_pipe_gridsearch, \
    make_preprocessing_pipeline_for_cv_with_function, PredictionInput, TransferModelInput, ExplanationInput
from modelling.data_access import InsideAccess

import logging


class Model(InsideAccess):

    def __init__(self):
        super().__init__()
        self.key_name_in_dataframes = None
        self.second_key_name_in_dataframes = None
        self.name_of_target = None
        self.train_set_row_index = None
        self.test_set_row_index = None
        self.X = None
        self.X_train = None
        self.X_test = None
        self.y = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.unfitted_estimator = None
        self.estimator_is_pipeline = False
        self.hyper_params = None
        self.group = None
        self.are_features_excluded = False
        self.features_in_training = None
        self._set_key_target_features_has_run = False
        self._set_estimators_and_parameters_has_run = False
        self._train_test_split_has_run = False

    def set_dataframe(self, dataframe: pd.DataFrame):
        check_parameter_is_of_type(parameter=dataframe, parameter_type=pd.DataFrame)
        self.dataframe = dataframe

    def set_key_target_features(self, key: str, target: str, features: list = None,
                                features_excluded_in_training: List[str] = None, second_key: str = None):
        self._check_dataframe_is_set()
        self._check_key_is_in_dataframe(key)
        check_parameter_is_of_type(parameter=key, parameter_type=str)
        self.key_name_in_dataframes = key
        check_parameter_is_of_type(parameter=target, parameter_type=str)
        self.name_of_target = target
        if features is None:
            features = [feat for feat in self.dataframe.columns if feat is not self.name_of_target]
        else:
            check_parameter_is_of_type(parameter=features, parameter_type=list)
            features = [feat for feat in features if feat is not self.name_of_target]
        self.X = self.dataframe.loc[:, features]
        self.y = self.dataframe.loc[:, target]
        if features_excluded_in_training is not None:
            check_parameter_is_of_type(parameter=features_excluded_in_training, parameter_type=list)
            self.are_features_excluded = True
            self.features_in_training = [feat for feat in features if feat not in features_excluded_in_training]
        else:
            self.features_in_training = features
        if second_key is not None:
            check_parameter_is_of_type(parameter=second_key, parameter_type=str)
            self.second_key_name_in_dataframes = second_key
        self._set_key_target_features_has_run = True

    def set_estimator_and_parameters(self, unfitted_estimator, hyper_params_dict_name_in_config: str):
        hyp_params = eval(self.configuration.get_config_setting(config_section='Model.Hyperparams',
                                                                config_name=hyper_params_dict_name_in_config))
        unfit_est = unfitted_estimator
        if unfit_est.__class__.__name__ == 'LogisticRegression':
            hyp_params = make_param_grid_for_pipe_gridsearch('estimator', hyp_params)
            unfit_est = self._make_pipeline(unfitted_estimator=unfit_est)
            self.estimator_is_pipeline = True
        self.hyper_params = hyp_params
        self.unfitted_estimator = unfit_est
        self._set_estimators_and_parameters_has_run = True

    def train_test_split_dataframe(self, group: list or str = None):
        self._check_set_key_target_features_has_run()
        if self.X is not None and self.y is not None:
            test_set_size = float(
                self.configuration.get_config_setting(config_section='Model.General', config_name='test_set_size'))
            if group is not None:
                self.group = group
                X_group = self.X.loc[:, self.group]
            else:
                X_group = None
            self.train_set_row_index, self.test_set_row_index = next(
                GroupShuffleSplit(n_splits=1, test_size=test_set_size, random_state=None).split(X=self.X, y=self.y,
                                                                                                groups=X_group))
            self._set_train_test_set_rows()
        else:
            raise ValueError('Features and target not set yet. Set them first before splitting into train-/test-sets !')
        self._train_test_split_has_run = True

    def train_model_and_cross_validate(self):
        self._check_dataframe_is_set()
        self._check_set_key_target_features_has_run()
        self._check_set_estimators_and_parameters_has_run()
        self._check_train_test_split_has_run()
        if all(attr is not None for attr in
               [self.X_train, self.y_train, self.unfitted_estimator, self.train_set_row_index]):
            if self.group is not None:
                X_train_group = self.X_train.loc[:, self.group]
            else:
                X_train_group = None
            num_cv_folds = int(
                self.configuration.get_config_setting(config_section='Model.General', config_name='num_cv_folds'))
            grid_search_iterator = GroupKFold(n_splits=num_cv_folds).split(X=self.X_train, y=self.y_train,
                                                                           groups=X_train_group)
            scoring = eval(self.configuration.get_config_setting(config_section='Model.General', config_name='scoring'))
            grid_search = GridSearchCV(estimator=self.unfitted_estimator, param_grid=self.hyper_params, scoring=scoring,
                                       n_jobs=None,
                                       cv=grid_search_iterator, refit=scoring[0], return_train_score=True, verbose=0)
            self._exclude_features_in_train_test_set()
            self.model = grid_search.fit(self.X_train, self.y_train)
            self.fitted_estimator = self.model.best_estimator_
        else:
            raise ValueError(
                'Missing input for either X, y, train_set_row_index and/or estimator. Please set those first !')

    def under_sample_dataframe(self, sampling_strategy: str or float):
        if not self._train_test_split_has_run:
            under_sampler = RandomUnderSampler(sampling_strategy=sampling_strategy)
            self.X, self.y = under_sampler.fit_resample(self.X, self.y)
        else:
            raise TypeError('train_test_split_dataframe()-method has already run. sampling not possible anymore !')

    def save_test_set(self, config_section: str, model_type: str = None):
        self.target_file_path = self.configuration.get_path(config_section, 'X_test', model_type)
        self._stored_dataframe = self.X_test
        self.target_file_path = self.configuration.get_path(config_section, 'y_test', model_type)
        self._stored_dataframe = self.y_test

    def get_cv_results(self) -> pd.DataFrame:
        return pd.DataFrame(data=self.model.cv_results_).T

    def get_test_set_roc_auc_score(self) -> float:
        return roc_auc_score(self.y_test, self.fitted_estimator.predict_proba(self.X_test)[:, 1])

    def generate_prediction_input(self):
        self.prediction_input_object = PredictionInput(key_name_in_dataframes=self.key_name_in_dataframes,
                                                       model_best_estimator=self.fitted_estimator,
                                                       estimator_is_pipeline=self.estimator_is_pipeline,
                                                       features_in_training=self.features_in_training,
                                                       name_of_target=self.name_of_target)

    def generate_explanation_input(self):
        self.explanation_input_object = ExplanationInput(model_best_estimator=self.fitted_estimator,
                                                         X=self.X,
                                                         X_train=self.X_train,
                                                         X_test=self.X_test,
                                                         key_name_in_dataframes=self.key_name_in_dataframes,
                                                         second_key_name_in_dataframes=self.second_key_name_in_dataframes)

    def _set_train_test_set_rows(self):
        self.X_train = self.X.iloc[self.train_set_row_index, :]
        self.y_train = self.y.iloc[self.train_set_row_index]
        self.X_test = self.X.iloc[self.test_set_row_index, :]
        self.y_test = self.y.iloc[self.test_set_row_index]

    def _exclude_features_in_train_test_set(self):
        if self.are_features_excluded:
            self.X_train = self.X_train.loc[:, self.features_in_training]
            self.X_test = self.X_test.loc[:, self.features_in_training]

    def _check_dataframe_is_set(self):
        if self.dataframe is None:
            return FileNotFoundError('dataframe is not set yet !')

    def _check_train_test_split_has_run(self):
        if self._train_test_split_has_run is False:
            raise TypeError('Run train_test_split_dataframe() first !')

    def _check_set_key_target_features_has_run(self):
        if self._set_key_target_features_has_run is False:
            raise TypeError('Run set_key_target_features() method first !')

    def _check_set_estimators_and_parameters_has_run(self):
        if self._set_estimators_and_parameters_has_run is False:
            raise TypeError('Run set_estimators_and_parameters_has_run() method first !')

    def _check_key_is_in_dataframe(self, key: str):
        if key not in self.dataframe.columns:
            return ValueError('key is not in dataframe !')

    def _make_pipeline(self, unfitted_estimator):
        all_columns = self.features_in_training
        impute_dict = eval(self.configuration.get_config_setting(config_section='Model.CV.Preprocessing',
                                                                 config_name='impute'))
        winsorize_dict = eval(self.configuration.get_config_setting(config_section='Model.CV.Preprocessing',
                                                                    config_name='winsorize'))
        scale_dict = eval(self.configuration.get_config_setting(config_section='Model.CV.Preprocessing',
                                                                config_name='scale'))
        imp_cols = all_columns if impute_dict.get('impute_columns') == 'All' else impute_dict.get('impute_columns')

        imp_str = impute_dict.get('impute_strategy')
        imp_knn_num_neigh = impute_dict.get('impute_knn_num_neighbors')
        win_cols = all_columns if winsorize_dict.get('winsorize_columns') == 'All' else winsorize_dict.get(
            'winsorize_columns')
        win_lower = winsorize_dict.get('winsorize_lower_bound')
        win_upper = winsorize_dict.get('winsorize_upper_bound')
        scale_cols = all_columns if scale_dict.get('scale_columns') == 'All' else scale_dict.get('scale_columns')
        scale_str = scale_dict.get('scale_strategy')
        scale_min_max = scale_dict.get('scale_min_max_range')
        scale_rob_range = scale_dict.get('scale_robust_quantile_range')
        pipe = make_preprocessing_pipeline_for_cv_with_function(
            unfitted_estimator=unfitted_estimator,
            impute_columns=imp_cols,
            winsorize_columns=win_cols,
            scale_columns=scale_cols,
            impute_strategy=imp_str,
            impute_knn_num_neighbors=imp_knn_num_neigh,
            scale_strategy=scale_str,
            scale_min_max_range=scale_min_max,
            scale_robust_quantile_range=scale_rob_range,
            winsorize_lower_bound=win_lower,
            winsorize_upper_bound=win_upper)
        return pipe


class TransferModel(Model):

    def __init__(self):
        super().__init__()
        self.transfer_model_targets = None
        self.cutoff_columns = None
        self._set_transfer_model_inputs_has_run = False

    def set_transfer_model_inputs(self, transfer_model_input: TransferModelInput):
        check_parameter_is_of_type(parameter=transfer_model_input, parameter_type=TransferModelInput)
        check_parameter_is_of_type(parameter=transfer_model_input.name_of_target, parameter_type=str)
        check_parameter_is_of_type(parameter=transfer_model_input.transfer_model_targets, parameter_type=pd.DataFrame)
        logging.warning(
            f" Please be aware that the dataframe of the TransferModel MUST NOT CONTAIN the target column as the"
            f" target column is generated within the transfer model itself. Please check !")
        # transfer_model_targets contain the columns: key (i.e. "orig_key_1") and quantiles from all models
        self.name_of_target = 'TRANS_TARGET_of_' + transfer_model_input.name_of_target
        self.transfer_model_targets = transfer_model_input.transfer_model_targets  # contains "key" at first column
        self.key_name_in_dataframes = transfer_model_input.key_name_in_dataframes
        self._create_target_column()
        self._combine_targets_with_features()
        self._set_transfer_model_inputs_has_run = True

    def set_key_target_features(self, key: str = None, target: str = None, features: list = None,
                                features_excluded_in_training: List[str] = None, second_key: str = None):
        # We must invalidate the parameters "key", "second_key" and "target" as they are already set with
        # the set_transfer_model_inputs()-method
        self._check_set_transfer_model_inputs_has_run()
        self._check_dataframe_is_set()
        super().set_key_target_features(key=self.key_name_in_dataframes,
                                        target=self.name_of_target,
                                        features=features,
                                        features_excluded_in_training=features_excluded_in_training,
                                        second_key=self.second_key_name_in_dataframes)

    def _create_target_column(self):
        estimators_for_cutoff = eval(self.configuration.get_config_setting(config_section='Transfer.Model.Cutoff',
                                                                           config_name='estimators_for_cutoff'))
        cutoff_threshold = float(self.configuration.get_config_setting(config_section='Transfer.Model.Cutoff',
                                                                       config_name='cutoff_threshold'))
        self.cutoff_columns = [name for name in self.transfer_model_targets.columns if
                               any(name_part in name for name_part in estimators_for_cutoff)]
        self.transfer_model_targets[self.name_of_target] = np.where(
            (self.transfer_model_targets[self.cutoff_columns] >= cutoff_threshold).all(axis='columns'), 1, 0)

    def _combine_targets_with_features(self):
        num_rows_before_merge = len(self.dataframe.index)
        key_and_target_columns = self.transfer_model_targets.loc[:, [self.key_name_in_dataframes, self.name_of_target]]
        self.dataframe = pd.merge(self.dataframe, key_and_target_columns, on=self.key_name_in_dataframes,
                                  how='inner')
        num_rows_after_merge = len(self.dataframe.index)
        if num_rows_before_merge != num_rows_after_merge:
            logging.warning(
                f" Please be aware that the number of dataframe rows was reduced from {num_rows_before_merge!s}"
                f" rows to {num_rows_after_merge!s} rows as the targets did not contain all"
                f" '{self.key_name_in_dataframes}'s that were in the dataframe!")

    def _check_set_transfer_model_inputs_has_run(self):
        if not self._set_transfer_model_inputs_has_run:
            raise ValueError('Run set_transfer_model_inputs() method first !')
