import pandas as pd
import numpy as np
import shap
import random
import matplotlib.pyplot as plt

from modelling.data_access import InsideAccess
from modelling.functions import check_parameter_is_of_type, ExplanationInput


class Explanation(InsideAccess):

    def __init__(self, explanation_input: ExplanationInput):
        check_parameter_is_of_type(parameter=explanation_input, parameter_type=ExplanationInput)
        super().__init__()
        self.model_estimator = explanation_input.model_best_estimator
        self.X = explanation_input.X
        self.X_train = explanation_input.X_train
        self.X_test = explanation_input.X_test
        self.key_name_in_dataframes = explanation_input.key_name_in_dataframes
        self.second_key_name_in_dataframes = explanation_input.second_key_name_in_dataframes
        self.keys = None
        self.key_values = None


class Shap(Explanation):
    def __init__(self, explanation_input: ExplanationInput):
        super().__init__(explanation_input=explanation_input)
        self.shap_explainer_object = None
        self.shap_explanation_object = None
        self.shap_explanation_data = None
        self.shap_explanation_data_all = None
        self.global_explanation_object = None
        self.global_base_value = None
        self.global_shap_values = None
        self.shap_data_set = None
        self._create_shap_objects_has_run = False
        self._calc_global_explanation_has_run = False

    def create_shap_objects(self):
        self._get_shap_explanation_data()
        self._calc_keys()
        self._calc_key_values()
        self._reduce_shap_explanation_data_rows()
        # Create SHAP-Explainer and SHAP-Explanation objects
        """ As of May 2021, the scikit-learn-method "predict_proba" is not yet supported for estimators other than
        XGBClassifier. The following error message occurs if "model_output" is set to "predict_proba" (Quote):
        "Exception: Unrecognized model_output parameter value: predict_proba! If model.predict_proba is a
        valid function open a github issue to ask that this method be supported. If you want 'predict_proba'
        just use 'probability' for now " (Quote end). Thus: Instead of "predict_proba" I use "probability" in
         case the estimator is not XGBClassifier. """
        if self.model_estimator.__class__.__name__ == 'XGBClassifier':
            model_output = 'predict_proba'
        else:
            model_output = 'probability'
        shap_background_data_num_samples = int(self.configuration.get_config_setting(config_section='Model.Explanation',
                                                                                     config_name='shap_background_data_num_samples'))
        # "data" passed to create the explainer object must be a masker instead of the actual dataframe, because
        # otherwise the standard masker is set in the background whose "max_samples"-attribute is set to 100 rows only.
        masker = shap.maskers.Independent(data=self.X_train,
                                          max_samples=shap_background_data_num_samples)
        self.shap_explainer_object = shap.explainers.Tree(model=self.model_estimator,
                                                          data=masker,
                                                          feature_names=self.X_train.columns.to_list(),
                                                          feature_perturbation="interventional",
                                                          model_output=model_output)
        self.shap_explanation_object = self.shap_explainer_object(self.shap_explanation_data)
        self._create_shap_objects_has_run = True

    def calc_global_explanation(self, target_class: int = 1):
        self._check_create_shap_objects_has_run()
        check_parameter_is_of_type(parameter=target_class, parameter_type=int)
        # Define global variables for class = target class (usually the target class is 1 (or in seldom cases 0))
        self.global_explanation_object = self.shap_explanation_object[:, :, target_class]
        self.global_base_value = self.shap_explainer_object.expected_value[target_class]
        self.global_shap_values = pd.DataFrame(data=self.shap_explanation_object.values[:, :, target_class],
                                               index=self.shap_explanation_data.index,
                                               columns=self.shap_explanation_data.columns)
        self._calc_global_explanation_has_run = True

    def _check_index_is_in_shap_explanation_data(self, key_value, second_key_value=None):
        key_value_index = self._get_index_of_key_value(key_value=key_value, second_key_value=second_key_value)
        if key_value_index not in self.shap_explanation_data.index:
            raise ValueError(f'KEY ERROR: The {self.key_name_in_dataframes} = {key_value!s} cannot be found in the '
                             f'shap_explanation_data and thus also not in the global_shap_values. The '
                             f'{self.key_name_in_dataframes} = {key_value!s} was probably excluded when the data size '
                             f'was reduced to the size given in the config-file as "shap_max_num_explanation_data_rows" !')

    def _check_index_is_in_shap_explanation_data_all(self, key_value, second_key_value=None):
        key_value_index = self._get_index_of_key_value(key_value=key_value, second_key_value=second_key_value)
        if key_value_index not in self.shap_explanation_data_all.index:
            raise ValueError(f'The {self.key_name_in_dataframes} = {key_value} could not be found in the '
                             f'provided data set "{self.shap_data_set}" !')

    def get_local_shap_values_from_global(self, key_value, second_key_value=None) -> np.ndarray:
        self._check_create_shap_objects_has_run()
        self._check_calc_global_explanation_has_run()
        self._check_index_is_in_shap_explanation_data(key_value=key_value, second_key_value=second_key_value)
        key_value_index = self._get_index_of_key_value(key_value=key_value, second_key_value=second_key_value)
        return self.global_shap_values[self.global_shap_values.index == key_value_index]

    def get_local_shap_values_from_method(self, key_value, second_key_value=None,
                                          target_class: int = 1) -> pd.DataFrame:
        self._check_create_shap_objects_has_run()
        self._check_calc_global_explanation_has_run()
        self._check_index_is_in_shap_explanation_data_all(key_value=key_value, second_key_value=second_key_value)
        key_value_index = self._get_index_of_key_value(key_value=key_value, second_key_value=second_key_value)
        data_row_to_calc_shap_for = self.shap_explanation_data_all[
            self.shap_explanation_data_all.index == key_value_index].to_numpy()
        data_row_shap_values = self.shap_explainer_object.shap_values(data_row_to_calc_shap_for)[target_class]
        return pd.DataFrame(data=data_row_shap_values,
                            index=self.shap_explanation_data_all.index[
                                self.shap_explanation_data_all.index == key_value_index],
                            columns=self.shap_explanation_data_all.columns)

    def compare_local_shap_from_global_with_local_shap_from_method(self, key_value, second_key_value=None,
                                                                   target_class: int = 1) -> pd.Series or pd.DataFrame:
        """ As also noted in the Notebook files in "FRAUD" (old), there is an issue with the local shap values as there
        is a difference between local shap values coming from global and local shap values coming from the method. With
        this method, the difference can be shown. The error should be corrected in newer versions of shap. """
        local_from_global = self.get_local_shap_values_from_global(key_value=key_value,
                                                                   second_key_value=second_key_value)
        local_from_method = self.get_local_shap_values_from_method(key_value=key_value,
                                                                   second_key_value=second_key_value,
                                                                   target_class=target_class)
        if local_from_global.shape == local_from_method.shape:
            print('The difference in shap-values between local_from_method and local_from_global is shown here.'
                  'The SHAP-value for the following feature(s) should be higher(+)/lower(-) by this amount in the '
                  'local_from_global-Plots: ')
            diff_method_global = (local_from_method - local_from_global).T
            return diff_method_global[diff_method_global.values != 0]
        else:
            return f'The DataFrames are different in their shape: Shape of local_from_global is ' \
                   f'{local_from_global.shape} and shape of local_from_method is {local_from_method.shape}. ' \
                   f'They thus cannot be compared !'

    def plot_global_bars(self, num_feat_shown: int = 12, save_plot_as_pdf: bool = False):
        self._check_create_shap_objects_has_run()
        self._check_calc_global_explanation_has_run()
        shap.plots.bar(shap_values=self.global_explanation_object, max_display=num_feat_shown, show=False)
        if save_plot_as_pdf:
            plot_name = 'global_bar_plot_' + self.model_estimator.__class__.__name__ + '.pdf'
            plt.savefig(plot_name, format='pdf', dpi=1200, bbox_inches='tight')

    def plot_global_beeswarm(self, num_feat_shown: int = 12, save_plot_as_pdf: bool = False):
        self._check_create_shap_objects_has_run()
        self._check_calc_global_explanation_has_run()
        shap.plots.beeswarm(shap_values=self.global_explanation_object, max_display=num_feat_shown, show=False)
        if save_plot_as_pdf:
            plot_name = 'global_beeswarm_plot_' + self.model_estimator.__class__.__name__ + '.pdf'
            plt.savefig(plot_name, format='pdf', dpi=1200, bbox_inches='tight')

    def plot_global_heatmap(self, num_feat_shown: int = 12, save_plot_as_pdf: bool = False):
        self._check_create_shap_objects_has_run()
        self._check_calc_global_explanation_has_run()
        shap.plots.heatmap(shap_values=self.global_explanation_object, max_display=num_feat_shown)
        if save_plot_as_pdf:
            plot_name = 'global_heatmap_plot_' + self.model_estimator.__class__.__name__ + '.pdf'
            plt.savefig(plot_name, format='pdf', dpi=1200, bbox_inches='tight')

    def plot_global_scatter_for_features(self, feature_name_one: str,
                                         display_only_feat_one: bool = True,
                                         feature_name_two: str = None,
                                         save_plot_as_pdf: bool = False):
        """ Adjusted quote from the SHAP website: "If 'feature_name_two' is 'None' and 'display_only_feat_one' is
        'True', this plot only scatters the SHAP values for 'feature_name_one'. If 'feature_name_two' is 'None' and
        'display_only_feat_one' is 'False', then the scatter plot points are colored by the feature that seems to
        have the strongest interaction effect with the first feature. If 'feature_name_two' is another feature name,
        then the scatter plot points are colored by this second feature." """
        self._check_create_shap_objects_has_run()
        self._check_calc_global_explanation_has_run()
        check_parameter_is_of_type(parameter=feature_name_one, parameter_type=str)
        if feature_name_two is None and display_only_feat_one:
            color = None
        elif feature_name_two is None and not display_only_feat_one:
            color = self.global_explanation_object
        else:
            check_parameter_is_of_type(parameter=feature_name_two, parameter_type=str)
            color = self.global_explanation_object[:, feature_name_two]
        shap.plots.scatter(shap_values=self.global_explanation_object[:, feature_name_one],
                           color=color,
                           show=False)
        if save_plot_as_pdf:
            plot_name = 'global_scatter_plot_' + self.model_estimator.__class__.__name__ + '.pdf'
            plt.savefig(plot_name, format='pdf', dpi=1200, bbox_inches='tight')

    def plot_local_force(self, key_value, second_key_value=None, target_class: int = 1,
                         contribution_threshold: float = 0.02,
                         save_plot_as_pdf: bool = False):
        local_shap_values = self.get_local_shap_values_from_method(key_value=key_value,
                                                                   target_class=target_class,
                                                                   second_key_value=second_key_value)
        key_value_index = self._get_index_of_key_value(key_value=key_value, second_key_value=second_key_value)
        shap.plots.force(base_value=np.around(self.global_base_value, decimals=2),
                         shap_values=np.around(local_shap_values.to_numpy(), decimals=2),
                         feature_names=self.shap_explanation_data_all.columns.to_list(),
                         features=np.around(self.shap_explanation_data_all[
                                                self.shap_explanation_data_all.index == key_value_index], decimals=2),
                         matplotlib=True, show=False, contribution_threshold=contribution_threshold)
        if save_plot_as_pdf:
            plot_name = 'local_force_plot_' + self.model_estimator.__class__.__name__ + key_value + '.pdf'
            plt.savefig(plot_name, format='pdf', dpi=1200, bbox_inches='tight')

    def plot_local_bars_from_global(self, key_value, second_key_value=None,
                                    num_feat_shown: int = 12,
                                    save_plot_as_pdf: bool = False):
        local_explanation_object = self._get_local_explanation_object(key_value=key_value,
                                                                      second_key_value=second_key_value)
        shap.plots.bar(local_explanation_object, max_display=num_feat_shown, show=False)
        if save_plot_as_pdf:
            plot_name = 'local_bar_plot_' + self.model_estimator.__class__.__name__ + key_value + '.pdf'
            plt.savefig(plot_name, format='pdf', dpi=1200, bbox_inches='tight')

    def plot_local_waterfall_from_global(self, key_value, second_key_value=None,
                                         num_feat_shown: int = 12,
                                         save_plot_as_pdf: bool = False):
        local_explanation_object = self._get_local_explanation_object(key_value=key_value,
                                                                      second_key_value=second_key_value)
        shap.plots.waterfall(local_explanation_object, max_display=num_feat_shown)
        if save_plot_as_pdf:
            plot_name = 'local_waterfall_plot_' + self.model_estimator.__class__.__name__ + key_value + '.pdf'
            plt.savefig(plot_name, format='pdf', dpi=1200, bbox_inches='tight')

    def _get_local_explanation_object(self, key_value, second_key_value=None):
        self._check_create_shap_objects_has_run()
        self._check_calc_global_explanation_has_run()
        self._check_index_is_in_shap_explanation_data(key_value=key_value, second_key_value=second_key_value)
        key_value_index = int(self._get_index_of_key_value(key_value=key_value, second_key_value=second_key_value))
        row_number = self._get_row_number_of_global_shap_values(index=key_value_index)
        local_explanation_object = self.global_explanation_object[row_number, :]
        return local_explanation_object

    def get_info_about_key_vals(self, key_value, second_key_value=None):
        key_value_index = self._get_index_of_key_value(key_value=key_value, second_key_value=second_key_value)
        if second_key_value is not None:
            info_text_0 = f'INFO: The {self.key_name_in_dataframes} = {key_value} with ' \
                          f'{self.second_key_name_in_dataframes} = {second_key_value} '
        else:
            info_text_0 = f'INFO: The {self.key_name_in_dataframes} = {key_value} '
        if key_value_index in self.shap_explanation_data.index:
            info_text_1 = f'IS in the shap_explanation_data (reduced data set) '
        else:
            info_text_1 = f'IS NOT in the shap_explanation_data (reduced data set) '
        if key_value_index in self.shap_explanation_data_all.index:
            info_text_2 = f'and IS in the shap_explanation_data_all !'
        else:
            info_text_2 = f'and IS NOT in the shap_explanation_data_all !'
        if key_value_index not in self.shap_explanation_data.index and key_value_index not in self.shap_explanation_data_all.index:
            info_text_3 = f' Thus, the searched data is NOT in the provided "{self.shap_data_set}" data set !'
        else:
            info_text_3 = ''
        print(info_text_0 + info_text_1 + info_text_2 + info_text_3)

    def _get_index_of_key_value(self, key_value, second_key_value) -> int or ValueError:
        if self.second_key_name_in_dataframes is not None and second_key_value is None:
            raise FileNotFoundError(
                f'Please provide a parameter value for the second_key "{self.second_key_name_in_dataframes}" !')
        if self.second_key_name_in_dataframes is None and second_key_value is not None:
            raise TypeError(
                f'The parameter value for the second_key cannot be processed! '
                f'Please omit the parameter value {second_key_value} !')
        if second_key_value is None:
            condition = (self.key_values == key_value)
        else:
            condition = (self.key_values[self.key_name_in_dataframes] == key_value) & \
                        (self.key_values[self.second_key_name_in_dataframes] == second_key_value)
        if len(self.key_values[condition]) == 1:
            return self.key_values[condition].index[0]
        else:
            raise ValueError(
                f'Either the row values for {self.key_name_in_dataframes} = {key_value} and '
                f'{self.second_key_name_in_dataframes} = {second_key_value} were not found or the row index for the '
                f'first is different from the row index of the latter. Please check your entries !')

    def _get_row_number_of_global_shap_values(self, index: int) -> int:
        check_parameter_is_of_type(parameter=index, parameter_type=int)
        return np.where(self.global_shap_values.index == index)[0][0]

    def _get_shap_explanation_data(self):
        self.shap_data_set = self.configuration.get_config_setting(config_section='Model.Explanation',
                                                                   config_name='shap_data_for_explanation')
        if self.shap_data_set == 'train':
            self.shap_explanation_data = self.shap_explanation_data_all = self.X_train
        elif self.shap_data_set == 'test':
            self.shap_explanation_data = self.shap_explanation_data_all = self.X_test
        elif self.shap_data_set == 'monthly':
            self._check_data_for_prediction_is_set()
            self.shap_explanation_data = self.shap_explanation_data_all = self.data_for_prediction.loc[:,
                                                                          self.X_train.columns.tolist()]
        else:
            raise FileNotFoundError(f'ERROR: {self.shap_data_set} not found or it is not a valid shap_data_set name!')

    def _calc_keys(self):
        if self.second_key_name_in_dataframes is None:
            self.keys = self.key_name_in_dataframes
        else:
            self.keys = list((self.key_name_in_dataframes, self.second_key_name_in_dataframes))

    def _calc_key_values(self):
        if self.shap_data_set == 'monthly':
            self._check_data_for_prediction_is_set()
            self.key_values = self.data_for_prediction.loc[:, self.keys]
        else:
            self.key_values = self.X.loc[:, self.keys]

    def _reduce_shap_explanation_data_rows(self):
        max_num_data_rows_config = \
            int(self.configuration.get_config_setting(config_section='Model.Explanation',
                                                      config_name='shap_max_num_explanation_data_rows'))
        num_data_rows_in_shap_explanation_data = len(self.shap_explanation_data.index)
        num_data_rows = min(max_num_data_rows_config, num_data_rows_in_shap_explanation_data)
        if num_data_rows != num_data_rows_in_shap_explanation_data:
            print(f'The size of the shap_explanation_data is now reduced from {num_data_rows_in_shap_explanation_data}'
                  f' rows to {num_data_rows} rows !')
        randomized_rows_list = random.sample(self.shap_explanation_data.index.to_list(), num_data_rows)
        randomized_index = self.shap_explanation_data.index.isin(randomized_rows_list)
        self.shap_explanation_data = self.shap_explanation_data[randomized_index]

    def _check_data_for_prediction_is_set(self):
        if self.data_for_prediction is None:
            raise TypeError('"data_for_prediction" not set yet. Please load it first !')

    def _check_shap_dataset_is_set(self):
        if self.shap_explanation_data is None:
            raise TypeError('"shap_explanation_data" not set yet. Please run create_shap_objects()-method first !')

    def _check_create_shap_objects_has_run(self):
        if not self._create_shap_objects_has_run:
            raise TypeError('Run create_shap_objects() method first !')

    def _check_calc_global_explanation_has_run(self):
        if not self._calc_global_explanation_has_run:
            raise TypeError('Run calc_global_explanation() method first !')
