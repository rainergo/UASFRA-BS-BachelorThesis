import pandas as pd

from modelling.data_access import OutsideAccess
from modelling.functions import check_parameter_is_of_type, PredictionInput, TransferModelInput


class Prediction(OutsideAccess):
    quantile_columns = list()

    def __init__(self, prediction_input: PredictionInput):
        check_parameter_is_of_type(parameter=prediction_input, parameter_type=PredictionInput)
        super().__init__()
        self.model_estimator = prediction_input.model_best_estimator
        self.estimator_is_pipeline = prediction_input.estimator_is_pipeline
        self.X_train_test_columns = prediction_input.features_in_training
        self.key_name_in_dataframes = prediction_input.key_name_in_dataframes
        self.name_of_target = prediction_input.name_of_target
        self.instance_result_dataframe = None
        self.quantile_portfolio_keys = None
        self.instance_quantile_portfolio = None
        self.unique_quantiles = None
        self._calculate_quantiles_has_run = False
        self._predict_probas_has_run = False
        self._add_results_to_aggregate_has_run = False

    def predict_probas(self):
        self._check_prediction_dataframe_is_set()
        if self.instance_result_dataframe is None:
            self._instantiate_instance_result_dataframe()
        adjusted_data_for_prediction = self.prediction_dataframe.loc[:, self.X_train_test_columns]
        predict_probas = self.model_estimator.predict_proba(adjusted_data_for_prediction)[:, 1]
        pred_proba_column_name = self._get_pred_proba_column_name()
        self.instance_result_dataframe[pred_proba_column_name] = predict_probas
        self._predict_probas_has_run = True

    def set_quantile_keys(self, quantile_portfolio: pd.DataFrame = None):
        if quantile_portfolio is None:
            self._check_quantile_portfolio_is_set()
            self.quantile_portfolio_keys = self.quantile_portfolio[self.key_name_in_dataframes].tolist()
        else:
            check_parameter_is_of_type(parameter=quantile_portfolio, parameter_type=pd.DataFrame)
            self.quantile_portfolio_keys = quantile_portfolio[self.key_name_in_dataframes].tolist()

    def _check_quantile_portfolio_is_set(self):
        # This is the "Prediction" class attribute "quantile_portfolio"
        if self.quantile_portfolio is None:
            return FileNotFoundError('quantile_portfolio is not set yet !')

    def calculate_quantiles(self):
        self._check_instance_result_dataframe_is_set()
        self._check_predict_probas_has_run()
        pred_proba_column_name = self._get_pred_proba_column_name()
        # Transfer predict_probas() from monthly prediction data into the instance_quantile_portfolio
        self.instance_quantile_portfolio = \
            self.instance_result_dataframe[
                self.instance_result_dataframe[self.key_name_in_dataframes].isin(self.quantile_portfolio_keys)]
        num_bins = 1000
        # Calculate quantile bins in the instance_quantile_portfolio
        quantile_bin = pd.qcut(self.instance_quantile_portfolio[pred_proba_column_name], num_bins, labels=None,
                               duplicates='drop', precision=5)
        # Calculate quantiles in the instance_quantile_portfolio
        quantile = pd.qcut(self.instance_quantile_portfolio[pred_proba_column_name], num_bins, labels=False,
                           duplicates='drop',
                           precision=5) * 0.10
        bin_col_name = pred_proba_column_name + '_bin'
        quantile_col_name = pred_proba_column_name + '_quantile'
        # Store quantile_col_name for later usage in the transfer model. Storage must be at the
        # Prediction class level (not the instance level),  otherwise individual instances cannot be
        # aggregated consistently
        Prediction.quantile_columns.append(quantile_col_name)
        # Add bin and quantile values to the quantile_portfolio
        self.instance_quantile_portfolio[bin_col_name] = quantile_bin
        self.instance_quantile_portfolio[quantile_col_name] = quantile
        # Pack both (bin and quantile values) into a DataFrame and remove duplicates
        self.unique_quantiles = pd.DataFrame(data={bin_col_name: quantile_bin,
                                                   quantile_col_name: quantile}).drop_duplicates(inplace=False)
        # Allocate quantile bins from the instance_quantile_portfolio into the instance_result_dataframe i.e.
        # classify the predict_proba values in the instance_result_dataframe into the bins just calculated
        # in the instance_quantile_portfolio
        self.instance_result_dataframe[bin_col_name] = pd.cut(self.instance_result_dataframe[pred_proba_column_name],
                                                              bins=quantile_bin.cat.categories,
                                                              duplicates='drop', include_lowest=True, right=True,
                                                              precision=5)
        # Now also allocate the quantiles corresponding to the bins into the instance_result_dataframe
        self.instance_result_dataframe = pd.merge(self.instance_result_dataframe, self.unique_quantiles,
                                                  on=bin_col_name,
                                                  how='left')
        self._calculate_quantiles_has_run = True

    def get_pred_proba_for_key(self, key_value: str) -> str or KeyError:
        self._check_instance_result_dataframe_is_set()
        self._check_predict_probas_has_run()
        try:
            pred_proba_column_name = self._get_pred_proba_column_name()
            return \
                self.instance_result_dataframe[
                    self.instance_result_dataframe[self.key_name_in_dataframes] == key_value][
                    pred_proba_column_name].values[0]
        except KeyError:
            return f'"predict_proba" not found for {self.key_name_in_dataframes}: {key_value!s} !'

    def add_results_to_aggregate(self):
        # Here we have to aggregate at the OutsideAccess class level (not the Prediction instance level),
        # otherwise individual instances cannot be aggregated consistently and the results cannot be saved
        self._check_predict_probas_has_run()
        self._check_calculate_quantiles_has_run()
        try:
            if OutsideAccess.result_dataframe is None:
                OutsideAccess.result_dataframe = self.instance_result_dataframe
            else:
                OutsideAccess.result_dataframe = pd.merge(OutsideAccess.result_dataframe,
                                                          self.instance_result_dataframe,
                                                          on=self.key_name_in_dataframes,
                                                          how='left')
            if OutsideAccess.quantile_portfolio is None:
                OutsideAccess.quantile_portfolio = self.instance_quantile_portfolio
            else:
                OutsideAccess.quantile_portfolio = pd.merge(OutsideAccess.quantile_portfolio,
                                                            self.instance_quantile_portfolio,
                                                            on=self.key_name_in_dataframes,
                                                            how='left')
        except ValueError:
            print('Could not aggregate results. Check if instance and class dataframes have same shape !')
        self._add_results_to_aggregate_has_run = True

    def generate_transfer_model_inputs(self):
        # Here we get the (aggregated) OutsideAccess class attributes, as the instance attributes only
        # contain individual (not aggregated) instance results and quantiles
        self._check_add_results_to_aggregate_has_run()
        transfer_model_target_columns = self.quantile_columns.copy()
        transfer_model_target_columns.insert(0, self.key_name_in_dataframes)
        targets = self.result_dataframe.loc[:, transfer_model_target_columns]
        self.transfer_model_input_object = TransferModelInput(transfer_model_targets=targets,
                                                              name_of_target=self.name_of_target,
                                                              key_name_in_dataframes=self.key_name_in_dataframes)

    def _check_instance_result_dataframe_is_set(self):
        if self.instance_result_dataframe is None or \
                self.key_name_in_dataframes not in self.instance_result_dataframe.columns:
            return FileNotFoundError('instance_result_dataframe is not set yet or does not contain key column !')

    def _check_prediction_dataframe_is_set(self):
        if self.prediction_dataframe is None:
            return FileNotFoundError('prediction_dataframe is not set yet !')

    def _check_features_in_prediction_dataframe(self):
        feat_not_in = [feat for feat in self.X_train_test_columns if feat not in self.prediction_dataframe.columns]
        if len(feat_not_in) > 0 or self.key_name_in_dataframes not in self.prediction_dataframe.columns:
            raise TypeError(
                f'Either the required features {feat_not_in!r} or the key {self.key_name_in_dataframes}'
                f' are missing in the prediction_dataframe !')

    def _instantiate_instance_result_dataframe(self):
        self._check_prediction_dataframe_is_set()
        self._check_features_in_prediction_dataframe()
        self.instance_result_dataframe = pd.DataFrame(self.prediction_dataframe.loc[:, self.key_name_in_dataframes])

    def _check_calculate_quantiles_has_run(self):
        if not self._calculate_quantiles_has_run:
            raise ValueError('Run calculate_quantiles() method first !')

    def _check_predict_probas_has_run(self):
        if not self._predict_probas_has_run:
            raise ValueError('Run predict_probas() method first !')

    def _check_add_results_to_aggregate_has_run(self):
        if not self._add_results_to_aggregate_has_run:
            raise ValueError('Run add_results_to_aggregate() method first !')

    def _get_pred_proba_column_name(self):
        if self.estimator_is_pipeline:
            estimator_name = 'Pipeline_' + self.model_estimator.named_steps['estimator'].__class__.__name__
        else:
            estimator_name = self.model_estimator.__class__.__name__
        return 'predict_probas_' + estimator_name
