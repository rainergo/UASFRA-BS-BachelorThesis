import numpy as np
import pandas as pd
from collections import namedtuple
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler, FunctionTransformer, \
    PowerTransformer
from sklearn.impute import SimpleImputer, KNNImputer

PredictionInput = namedtuple('PredictionInput', (
    'key_name_in_dataframes', 'model_best_estimator', 'estimator_is_pipeline', 'features_in_training',
    'name_of_target'))
TransferModelInput = namedtuple('TransferModelInput',
                                ('transfer_model_targets', 'name_of_target', 'key_name_in_dataframes'))
ExplanationInput = namedtuple('ExplanationInput', (
    'model_best_estimator', 'X', 'X_train', 'X_test', 'key_name_in_dataframes',
    'second_key_name_in_dataframes'))


def check_parameter_is_of_type(parameter: any, parameter_type: object):
    error_message = f'Provided parameter {parameter!s} is not of required type {parameter_type!s}. Please check !'
    if isinstance(parameter_type, list):
        if not type(parameter) in parameter_type:
            raise TypeError(error_message)
    else:
        if not type(parameter) is parameter_type:
            raise TypeError(error_message)


def dataframe_col_name_to_num_list(dataframe: pd.DataFrame, col_names: list) -> list:
    check_parameter_is_of_type(parameter=dataframe, parameter_type=pd.DataFrame)
    check_parameter_is_of_type(parameter=col_names, parameter_type=list)
    return [dataframe.columns.get_loc(col) if (type(col) == str) else col for col in
            col_names]


def check_list_has_elements(parameter_list: list):
    if len(parameter_list) == 0:
        raise IndexError(f'Provided parameter "{parameter_list!s}" has no elements. Please check !')


def get_key_values_where_index_is_in_both_dataframes(df_one: pd.DataFrame, df_two: pd.DataFrame, key_in_df_one: str,
                                                     second_key_in_df_one: str = None) -> list:
    check_parameter_is_of_type(parameter=df_one, parameter_type=pd.DataFrame)
    check_parameter_is_of_type(parameter=df_two, parameter_type=pd.DataFrame)
    check_parameter_is_of_type(parameter=key_in_df_one, parameter_type=str)
    if second_key_in_df_one is None:
        return df_one[df_one.index.isin(df_two.index)][key_in_df_one].values.tolist()
    else:
        check_parameter_is_of_type(parameter=second_key_in_df_one, parameter_type=str)
        return df_one[df_one.index.isin(df_two.index)].loc[:, [key_in_df_one, second_key_in_df_one]].values.tolist()


def make_param_grid_for_pipe_gridsearch(name_of_step_in_pipe: str, est_params: dict,
                                        existent_pipe_param_grid: dict = None):
    # Parameters of pipelines can be set using ‘__’ plus the name of the parameter:
    parameter_grid = {}
    new_grid = {name_of_step_in_pipe + '__' + k: v for k, v in est_params.items()}
    if existent_pipe_param_grid:
        parameter_grid.update(existent_pipe_param_grid)
    parameter_grid.update(new_grid)
    return parameter_grid


def winsorize_for_function_transformer(np_array, lower_bound: float = 0.00, upper_bound: float = 1.00):
    # Columns are not passed here as they are passed in the transformer tuple.
    # As the transformer currently (March 2021) only works on numpy arrays and cannot iterate over columns,
    # we must adjust the winsorize-function here.
    lower = np.quantile(np_array, lower_bound, axis=0, interpolation='higher')
    upper = np.quantile(np_array, upper_bound, axis=0, interpolation='lower')
    np_array = np.clip(np_array, lower, upper)
    return np_array


def make_preprocessing_pipeline_for_cv_with_function(unfitted_estimator,
                                                     impute_columns: list = None, winsorize_columns: list = None,
                                                     scale_columns: list = None,
                                                     impute_strategy: str = 'median', impute_knn_num_neighbors: int = 3,
                                                     scale_strategy: str = 'standard',
                                                     scale_min_max_range: tuple = (0, 1),
                                                     scale_robust_quantile_range: tuple = (25, 75),
                                                     winsorize_lower_bound: float = 0.00,
                                                     winsorize_upper_bound: float = 1.00) -> Pipeline:
    # 'impute_columns' or 'scale_columns' - list either of column names (strings) or of column numbers (integers)
    # Possible impute_strategies = ['median', 'mean', 'most_frequent', 'constant', 'knn']
    # Make sets and set intersections
    impute_set = set(impute_columns) if impute_columns is not None else set()
    winsorize_set = set(winsorize_columns) if winsorize_columns is not None else set()
    scale_set = set(scale_columns) if scale_columns is not None else set()
    impute_winsorize_scale_set = impute_set & winsorize_set & scale_set
    impute_winsorize_scale_list = list(impute_winsorize_scale_set)
    impute_only_list = list(impute_set - winsorize_set - scale_set)
    scale_only_list = list(scale_set - impute_set - winsorize_set)
    winsorize_only_list = list(winsorize_set - impute_set - scale_set)
    impute_scale_list = list(impute_set.intersection(impute_set, scale_set) - impute_winsorize_scale_set)
    impute_winsorize_list = list(impute_set.intersection(winsorize_set) - impute_winsorize_scale_set)
    winsorize_scale_list = list(winsorize_set.intersection(scale_set) - impute_winsorize_scale_set)

    # No1: Imputers
    if impute_strategy == 'knn':
        impute_function = KNNImputer(n_neighbors=impute_knn_num_neighbors, missing_values=np.nan)
    else:
        impute_function = SimpleImputer(strategy=impute_strategy, copy=False, missing_values=np.nan)

    # No2: Winsorizer
    # Takes function, in this case the function winsorize_for_function_transformer() from above
    winsorize_function = FunctionTransformer(func=winsorize_for_function_transformer,
                                             kw_args={'lower_bound': winsorize_lower_bound,
                                                      'upper_bound': winsorize_upper_bound})

    # No3: Scalers
    scalers = {'standard': StandardScaler(copy=False, with_mean=True, with_std=True),
               'robust': RobustScaler(with_centering=True, with_scaling=True,
                                      quantile_range=scale_robust_quantile_range, copy=False, unit_variance=False),
               'maxabs': MaxAbsScaler(copy=False),
               'minmax': MinMaxScaler(feature_range=scale_min_max_range, copy=False, clip=False),
               'power': PowerTransformer(copy=False, method='yeo-johnson', standardize=True)}
    scale_function = scalers.get(scale_strategy, 'Error: The scale_function could not be found')

    # If more than one transformation shall be made within the same column, then sequential processing needs to
    # be ensured with a Pipeline instance. See here:
    # https://towardsdatascience.com/simplifying-machine-learning-model-development-with-columntransformer
    # -pipeline-f09ffb04ca6b
    preprocess_pipeline_steps = []
    if len(impute_winsorize_scale_list) > 0:
        impute_winsorize_scale = Pipeline(
            steps=[('impute', impute_function),
                   ('winsorize', winsorize_function),
                   ('scale', scale_function)])
        impute_winsorize_scale_tuple = ('impute_winsorize_scale', impute_winsorize_scale, impute_winsorize_scale_list)
        preprocess_pipeline_steps.append(impute_winsorize_scale_tuple)

    if len(impute_winsorize_list) > 0:
        impute_winsorize = Pipeline(
            steps=[('impute', impute_function),
                   ('winsorize', winsorize_function)])
        impute_winsorize_tuple = ('impute_winsorize', impute_winsorize, impute_winsorize_list)
        preprocess_pipeline_steps.append(impute_winsorize_tuple)

    if len(impute_scale_list) > 0:
        impute_scale = Pipeline(
            steps=[('impute', impute_function),
                   ('scale', scale_function)])
        impute_scale_tuple = ('impute_scale', impute_scale, impute_scale_list)
        preprocess_pipeline_steps.append(impute_scale_tuple)

    if len(winsorize_scale_list) > 0:
        winsorize_scale = Pipeline(
            steps=[('winsorize', winsorize_function),
                   ('scale', scale_function)])
        winsorize_scale_tuple = ('winsorize_scale', winsorize_scale, winsorize_scale_list)
        preprocess_pipeline_steps.append(winsorize_scale_tuple)

    if len(impute_only_list) > 0:
        impute_only_tuple = ('impute_only', impute_function, impute_only_list)
        preprocess_pipeline_steps.append(impute_only_tuple)

    if len(winsorize_only_list) > 0:
        winsorize_only_tuple = ('winsorize_only', winsorize_function, winsorize_only_list)
        preprocess_pipeline_steps.append(winsorize_only_tuple)

    if len(scale_only_list) > 0:
        scale_only_tuple = ('scale_only', scale_function, scale_only_list)
        preprocess_pipeline_steps.append(scale_only_tuple)

    preprocess_transformer = ColumnTransformer(transformers=preprocess_pipeline_steps, remainder='passthrough')

    if preprocess_transformer:
        return Pipeline(steps=[('preprocess_transformer', preprocess_transformer), ('estimator', unfitted_estimator)])
    else:
        raise ValueError(
            'The preprocess_transformer contains no data. preprocess_transformer is: '.format(preprocess_transformer))
